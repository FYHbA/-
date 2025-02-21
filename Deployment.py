# 本文件是模型的部署文件，创建模型调用接口。

from transformers import BertTokenizerFast
from fastapi import FastAPI
import torch
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dataset import create_dataset, create_loader
import ruamel_yaml as yaml


config = yaml.load(open('./configs/used.yaml', 'r'), Loader=yaml.Loader)
# 在模型加载时检查设备并移动模型到适当设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def text_input_adjust(text_input, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids]) - 1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in
                                input_ids_remove_SEP]  # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    return text_input
# 定义输入数据的格式
class InputData(BaseModel):
    input_values: list

# 加载模型
def load_model(model_path):
    model = torch.load(model_path, map_location=device)  # 指定加载到对应设备
    model.to(device)  # 确保模型在正确的设备上
    model.eval()  # 设置模型为评估模式
    return model

# 使用 lifespan 事件处理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时加载模型和分词器
    model_path = 'finally_used_model/BiC_HAMMER.pth'  # 替换为你的 .pth 文件路径
    save_path = "finally_used_model/my_local_tokenizer"
    app.state.model = load_model(model_path)
    app.state.tokenizer = BertTokenizerFast.from_pretrained(save_path)
    yield
    # 在应用关闭时清理资源（可选）
    app.state.model = None
    app.state.tokenizer = None

# 创建 FastAPI 应用
app = FastAPI(lifespan=lifespan)

# 定义推理接口
@app.post("/predict")
async def predict(data: InputData):
    data_list=data.input_values
    val_dataset = create_dataset(config,data_list)

    samplers = [None]

    val_loader = create_loader([val_dataset],
                               samplers,
                               batch_size=[config['batch_size_val']],
                               num_workers=[4],
                               is_trains=[False],
                               collate_fns=[None])[0]
    for i, (image, label, text, W, H) in enumerate(val_loader):
        # 创建一个空的列表来存储所有的预测值
        pred_labels = []
        # 将数据移动到指定设备（如 GPU）
        image = image.to(device, non_blocking=True)
        text = list(text)
        # 对文本进行分词处理
        text_input = app.state.tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        # 调整文本输入格式

        text_input = text_input_adjust(text_input, device)

        # 模型前向传播
        with torch.no_grad():
            logits_real_fake = app.state.model(
                image, label, text_input,  is_train=False
            )
            pred_acc = logits_real_fake.argmax(1)
        # 将预测结果添加到列表中
            pred_labels.extend(pred_acc.cpu().numpy().tolist())  # .cpu().numpy() 将其转换为可操作的形式

    return pred_labels




# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
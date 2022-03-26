| MNIST  | Server                                                       | Client                                     |
| ------ | :----------------------------------------------------------- | :----------------------------------------- |
| step 0 | 初始化参数(全局迭代次数T，BatchSize，学习率lr，输出维度out_dim) | 接收初始化参数，初始化模型model            |
| step 1 | 生成下标idx，按照下标抽取样本的labels，                      | 按照下标抽取样本特征x                      |
| step 2 | - 等待Client...                                              | 计算预测值pred = model(x)，发送pred        |
| step 3 | 接收preds = sum(pred1,pred2,...,predn)；<br />根据preds预测得到pred_labels，与labels对比得到精度acc；<br />计算损失loss = CrossEntropyLoss(preds, labels)，将loss发送给Client | - 等待Server...                            |
| step 4 |                                                              | 接收loss，反向传播，更新模型权重           |
| step 5 | if 训练过程结束: 退出<br />else: 返回step1                   | if 训练过程结束: 退出<br />else: 返回step1 |
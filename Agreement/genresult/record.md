/*
prompt
    原始 
    /sum
    wx(pcf) 原始
    wx(average) 原始
    wx(pcf) /sum
    wx(average) / sum

    wx pcf / 10 sum

output 0bias 1bias 总bias
       0count 1conunt 总count
*/

想要解决的问题是：

句子更换主语后（男比女聪明->女比男聪明)， 理论上回答“对”的概率应该是和为100%的相反数，表明model识别出“A比B”这个结构，以此证明这种方法测量bias是有效果的；
实际应用中，无论是原始概率（model直接输出）还是5对verbalizer的归一化处理("是"和"否"的概率 / sum), 都无法体现出互补为100的特征；

使用了 Calibrate Before Use: Improving Few-Shot Performance of Language Models 论文中的方法，  用[MASK], N/A, "" 作为prompt求出verbalizer的平均概率pcf，然后计算变换矩阵W = Inv(I*pcf) ，y = Wx 得到新的输出概率。除了上述三个prompt外，另外尝试用了数据集中所有context作为prompt求出的平均概率pcf2；

效果是，用新的输出概率y得到的一些verbalizer pair中的bias极性变了：如[是的,不是]就从原来的严重bias(0.95:0.05)变为anti-bias(0.40:0.60)， 但是想要的“互补”效果仍无法体现，另外，用pcf2得到bias结果是更接近与0.5：0.5， 但是其他效果与pcf_origin一样

另外，用了pcf修正输出概率后，5对verbalizer一起做归一化( / SUM(5*2个词的概率)) 发现表示同意和否定的词仍然无法达到互补的效果

但是通过在t-检验 成对样本匹配，发现句子更换主语顺序后得到的两组output都存在显著差异，换句话说模型可以识别出context的不同，但是无法输出为我们想要的差异。

另外，在更换不同的prompt  (context, answer) -> (问题：context \n 答案: answer) 之后，上述现象仍然存在，但是个别verbalizer pair的概率会出现大幅改变，甚至极性相反。 说明model对特定词的概率输出很不稳定，受context的影响很大


解决的方法：通过收集数据和设计loss， finetune 大模型，使得大模型的输出达到“互补”的效果
成对数据输进去，

input_pair 是 [A比B聪明, B比A聪明]
output 是 归一化([是的, 不是] 的概率) (包括一系列的verbalizer_pair)
loss 是 abs(100-p1(是的) - p2(是的))

为什么要解决互补问题，因为不互补不知道模型真正的“态度”，
比如A比B聪明 0.4 B比A聪明 0.41 是否说明模型更偏向B呢？



试一下“A比B” “聪明”


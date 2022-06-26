# Intervention Result
|Model Name|Model Base Bias|Lower Bias|Upper Bias|Unrelated Bias|
|----|----|----|----|----|
|bert-base|0.2791|0.2208|0.0636|0.4952|
|bert-hfl-wwm-base|0.2482|0.1416|0.1244|0.5214|
|roberta-hfl-wwm-base|0.2494|0.2285|0.0494|0.5095|
|roberta-hfl-wwm-large|0.1845|0.019|0.0059|0.9244|
|ernie-baidu-base|0.2815|0.1154|0.0577|0.5922|
|albert-ckiplab-tiny|0.1333|0.0|0.0|0.9952|

Model Base Bias与SS测量相似，可以看出 roberta-hfl-wwm-large 和 albert的Model BaseBias比例明显偏低，与SS结果类似，但CrowsP本身与这两者不相似


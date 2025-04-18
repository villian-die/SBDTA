代码有三个地方需要手动调整，首先是数据预处理的preprocessing.py里面需要修改，现有的是针对KABA，Davis数据集里面的，如需要修改，可以将代码中的与备注（“# ”或者“ ''' ”里相似的代码进行互换就可以了，其次是trainer.py，里面也是一样进行修改，最后一个是main-model.py里面的cross_entropy_logits(linear_output, label)的方法

torch                        1.12.1          py3.8_cuda11.3_cudnn8.3.2_0
numpy                     1.23.5
scikit-learn              1.2.2
pandas                     1.4.1
rdkit                         2021.03.2
pytorch-geometric   2.3.0           py38_torch_1.12.0_cu113

运行方法如下：
1： python preprocessing.py
2： python main.py --cfg "config/Drug-DTA.yaml" --data ${dataset}
${dataset}可以填写KABA\Davis\human\biosnap\bindingdb
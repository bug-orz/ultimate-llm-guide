# Table of contents

* [大语言模型的终极之路](README.md)
* [更新计划](geng-xin-ji-hua.md)
* [大语言模型时代的NLP](new-era/README.md)
  * [任务与评测](new-era/ren-wu-yu-ping-ce.md)
  * [参考资料](new-era/can-kao-zi-liao.md)

## 基础知识 <a href="#basic" id="basic"></a>

* [负对数似然](basic/fu-dui-shu-si-ran.md)
* [Transformer](basic/transformer/README.md)
  * [Layer Normalization](basic/transformer/layer-normalization.md)
  * [Attention Block](basic/transformer/attention-block.md)
* [Page](basic/page.md)
* [优化算法](basic/you-hua-suan-fa.md)

## 大语言模型 <a href="#llm" id="llm"></a>

* [大模型理论](llm/da-mo-xing-li-lun/README.md)
  * [Scaling Law](llm/da-mo-xing-li-lun/scaling-law.md)
  * [The Bitter Lesson](llm/da-mo-xing-li-lun/the-bitter-lesson.md)
  * [思考，快与慢](llm/da-mo-xing-li-lun/si-kao-kuai-yu-man.md)
* [模型结构](llm/mo-xing-jie-gou/README.md)
  * [MLP](llm/mo-xing-jie-gou/mlp.md)
  * [Rotary Embedding](llm/mo-xing-jie-gou/rotary-embedding.md)
  * [RMSNorm](llm/mo-xing-jie-gou/rmsnorm.md)
  * [Encoder-decoder](llm/mo-xing-jie-gou/encoder-decoder.md)
  * [Decoder-only](llm/mo-xing-jie-gou/decoder-only.md)
  * [MOE](llm/mo-xing-jie-gou/moe.md)
  * [常见大模型](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/README.md)
    * [T5](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/t5.md)
    * [GPT2](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/gpt2.md)
    * [LLaMA](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/llama.md)
    * [LLaMA 2](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/llama-2.md)
    * [Mistral](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/mistral.md)
    * [GLM](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/glm.md)
    * [Mixture](llm/mo-xing-jie-gou/chang-jian-da-mo-xing/mixture.md)
* [如何训练一个ChatGPT](llm/ru-he-xun-lian-yi-ge-chatgpt.md)
* [微调](llm/wei-tiao/README.md)
  * [Instruction Tuning 指令微调](llm/wei-tiao/instruction-tuning-zhi-ling-wei-tiao.md)
  * [Domain Finetune 领域微调](llm/wei-tiao/domain-finetune-ling-yu-wei-tiao.md)
* [解码](llm/jie-ma.md)

## Prompt 工程 <a href="#prompt" id="prompt"></a>

* [Prompt， 一种技术路线](prompt/prompt-yi-zhong-ji-shu-lu-xian.md)
* [Prompt 写作规范](prompt/prompt-xie-zuo-gui-fan.md)
* [In-Context Learning](prompt/in-context-learning.md)
* [Chain-of-Thought](prompt/chain-of-thought.md)
* [Generate Rather than Read](prompt/generate-rather-than-read.md)
* [Program-of-Thought](prompt/program-of-thought.md)
* [Tree-of-Thought](prompt/tree-of-thought.md)
* [参考资料](prompt/can-kao-zi-liao.md)

***

* [知识与幻觉](zhi-shi-yu-huan-jue/README.md)
  * [知识边界](zhi-shi-yu-huan-jue/zhi-shi-bian-jie.md)

## 大规模预训练

* [计算资源消耗](da-gui-mo-yu-xun-lian/ji-suan-zi-yuan-xiao-hao.md)
* [Deepspeed](da-gui-mo-yu-xun-lian/deepspeed.md)
* [Megatron](da-gui-mo-yu-xun-lian/megatron.md)
* [大规模数据处理](da-gui-mo-yu-xun-lian/da-gui-mo-shu-ju-chu-li.md)
* [CUDA 算子优化](da-gui-mo-yu-xun-lian/cuda-suan-zi-you-hua.md)

## 强化学习 <a href="#reinforcement-learning" id="reinforcement-learning"></a>

* [RLHF](reinforcement-learning/rlhf.md)

## 大模型轻量化

* [蒸馏](da-mo-xing-qing-liang-hua/zheng-liu/README.md)
  * [黑盒蒸馏](da-mo-xing-qing-liang-hua/zheng-liu/hei-he-zheng-liu.md)
  * [白盒蒸馏](da-mo-xing-qing-liang-hua/zheng-liu/bai-he-zheng-liu/README.md)
    * [KL 散度](da-mo-xing-qing-liang-hua/zheng-liu/bai-he-zheng-liu/kl-san-du.md)
* [轻量化微调](da-mo-xing-qing-liang-hua/qing-liang-hua-wei-tiao/README.md)
  * [LoRA](da-mo-xing-qing-liang-hua/qing-liang-hua-wei-tiao/lora.md)
* [量化](da-mo-xing-qing-liang-hua/liang-hua.md)
* [剪枝](da-mo-xing-qing-liang-hua/jian-zhi.md)
* [推理加速](da-mo-xing-qing-liang-hua/tui-li-jia-su.md)
* [参考资料](da-mo-xing-qing-liang-hua/can-kao-zi-liao.md)

## RAG-大模型检索 <a href="#rag" id="rag"></a>

* [Page 3](rag/page-3.md)

## 多智能体 <a href="#agent" id="agent"></a>

* [Page 6](agent/page-6.md)

## 多模态大模型 <a href="#mllm" id="mllm"></a>

* [Page 1](mllm/page-1.md)

***

* [大模型安全与鲁棒](da-mo-xing-an-quan-yu-lu-bang.md)

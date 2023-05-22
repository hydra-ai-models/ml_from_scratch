# LLMs From Scratch

<p align="center">
  <img src="docs/images/readme_image.png" width=50%/>
</p>

Repository of architectures for Large Language Models (LLMs), along with algorithms for
data collection, pretraining, finetuning, evaluation and distillation.

All algorithms are implemented from scratch in PyTorch. Hence it serves as an excellent reference
for students, ML engineers and ML practitioners alike. This repository is inspired by the
minGPT (https://github.com/karpathy/minGPT) series, and adds on top by following industry best practices
like clean and well documented code, modular and well tested components, following standard PyTorch APIs
like Datasets and DataLoaders. The repository also aims to expand beyond pretraining, and include other algorithms too
like LoRA finetuning, distillation and RLHF alignment.

## Algorithms implemented in this repository
1. Pretraining of GPT models.
2. Finetuning of GPT models using LoRA.

## Getting started
To pretrain GPT model, run the python script `trainer_gpt_pretraining.py` using `python -m trainer_gpt_pretraining`, or run the Jupyter notebook `trainer_gpt_pretraining.ipynb`. Running this script will train the GPT model, and store the model file in the `output` directory. Similarly, a text file is created in this directory with more details about the trainable and non trainable parameters in the model.

To perform LoRA finetuning of the GPT model, run the python script `trainer_lora_finetuning.py` using `python -m trainer_lora_finetuning`, or run the Jupyter notebook `trainer_lora_finetuning.ipynb`. Running this script will finetuning the GPT model, and store the model file in the `output` directory. Similarly, a text file is created in this directory with more details about the trainable and non trainable parameters in the model.

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The views and opinions of authors expressed herein do not necessarily state or reflect those of their employers or any agency thereof.

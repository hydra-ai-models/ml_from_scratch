# Large Language Models

<p align="center">
  <img src="docs/images/readme_image.png" width=50%/>
</p>


## Getting started

### Setting up the dev environment for large language models.

Clone this repository and navigate to the `large_language_models` directory.

#### On Mac
Note that only small models with around 500k parameters can be trained on Mac due to limited memory and lack of GPUs. This is still a good step to do for prototyping architecture implementations on small datasets.

We will be using conda virtual environments for development. If you have not used conda before, install it first following instructions [here](https://developer.apple.com/metal/pytorch/). Now create a conda environment and activate it. Sample commands to run in the terminal are
`conda create --name llms_from_scratch_env python=3.9`

`conda activate llms_from_scratch_env`

Now install requirements by running `pip install -r requirements.txt`

#### On AWS
Create an [AWS account](https://aws.amazon.com/). Go to the EC2 page and click `Launch Instance`. Give a name for your instance, and select a recent PyTorch image as the AMI. Request a `t2.large` instance with 8GB RAM to try out the small example first. To scale to larger examples, you will need to request more powerful machines with GPUs. But it is a good idea to first get a small example working on a small machine to prototype. Click `Launch Instance` button to create the instance. Now click the instance, Connect -> SSH client, and copy the ssh command shown. Running this command from your mac will ssh into the EC2 instance. Once you ssh into the EC2 instance, clone this repository using `git clone git@github.com:hydra-ai-models/llms_from_scratch.git`. Now move to the main repository directory using `cd llms_from_scratch`.

Run the commands `conda create --name llms_from_scratch_env python=3.9`, `conda activate llms_from_scratch_env` and `pip install -r requirements.txt` as explained in the previous section to set up the dev environment.

### Running the trainer scripts
#### Without HuggingFace Accelerate
To pretrain GPT model, run the python script `trainer_gpt_pretraining.py` using `python -m trainer_gpt_pretraining`, or run the Jupyter notebook `trainer_gpt_pretraining.ipynb`. Running this script will train the GPT model, and store the model file in the `output` directory. Similarly, a text file is created in this directory with more details about the trainable and non trainable parameters in the model.

To perform LoRA finetuning of the GPT model, run the python script `trainer_lora_finetuning.py` using `python -m trainer_lora_finetuning`, or run the Jupyter notebook `trainer_lora_finetuning.ipynb`. Running this script will finetuning the GPT model, and store the model file in the `output` directory. Similarly, a text file is created in this directory with more details about the trainable and non trainable parameters in the model.

#### With HuggingFace Accelerate
[Hugging Face Accelerate](https://huggingface.co/docs/accelerate/main/en/index) allows you to scale training seamlessly in a distributed setting using optimizations from NVIDIA Megatron or DeepSpeed.
To use HuggingFace accelerate for training, first create a config file for training by running `accelerate config`. On Mac, set `No distributed training`, and answer No to `dynamo script optimization` and `mixed precision` training. Now run the training using `accelerate launch -m trainer.trainer_gpt_pretraining_with_accelerate`


## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The views and opinions of authors expressed herein do not necessarily state or reflect those of their employers or any agency thereof.

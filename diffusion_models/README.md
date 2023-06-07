# Diffusion Models

<p align="center">
  <img src="docs/images/image_diffusion_models.jpeg" width=50%/>
</p>


## Getting started

### Setting up the dev environment for diffusion models.

Clone this repository and navigate to the `diffusion_models` directory.

#### On Mac
We will be using conda virtual environments for development. If you have not used conda before, install it first following instructions [here](https://developer.apple.com/metal/pytorch/). Now create a conda environment and activate it. Sample commands to run in the terminal are
`conda create --name diffusion_models_from_scratch_env python=3.9`

`conda activate diffusion_models_from_scratch_env`

Now install requirements by running `pip install -r requirements.txt`

### Setting up the Hugging Face token.
We will be using Hugging Face datasets for training. To access these datasets, you have to be a registered user in 🤗 Hugging Face Hub, and you'll also need to use an access token for the code to work. For more information on access tokens, please refer to the [🤗 Hugging Face documentation](https://huggingface.co/docs/hub/security-tokens).

Run the following command to authenticate your token
`huggingface-cli login`

### Running the trainer scripts
To train the image diffusion model, run the python script `trainer_diffusion_model.py` using `python -m trainer.trainer_diffusion_model`, or run the Jupyter notebook `trainer_diffusion_model.ipynb`.

## Disclaimer

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The views and opinions of authors expressed herein do not necessarily state or reflect those of their employers or any agency thereof.

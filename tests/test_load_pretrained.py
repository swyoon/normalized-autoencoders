import pytest
from models import load_pretrained

# (identifier, config_file, ckpt_file, kwargs)
cifar_ood_nae = ('cifar_ood_nae/z32gn', 'z32gn.yml', 'nae_8.pkl', {})
cifar_ood_nae_ae = ('cifar_ood_nae/z32gn', 'z32gn.yml', 'model_best.pkl', {})  # NAE before training

mnist_ood_nae = ('mnist_ood_nae/z32', 'z32.yml', 'nae_20.pkl', {})
mnist_ood_nae_ae = ('mnist_ood_nae/z32', 'z32.yml', 'model_best.pkl', {})  # NAE before training

celeba64_ood_nae = ('celeba64_ood_nae/z64gr_h32g8', 'z64gr_h32g8.yml', 'nae_3.pkl', {})
celeba64_ood_nae_ae = ('celeba64_ood_nae/z64gr_h32g8', 'z64gr_h32g8.yml', 'model_best.pkl', {})  # NAE before training
l_setting = [cifar_ood_nae, cifar_ood_nae_ae,
             mnist_ood_nae, mnist_ood_nae_ae,
             celeba64_ood_nae, celeba64_ood_nae_ae] 


@pytest.mark.parametrize('model_setting', l_setting)
def test_load_pretrained(model_setting):
    identifier, config_file, ckpt_file, kwargs = model_setting
    model, cfg = load_pretrained(identifier, config_file, ckpt_file, **kwargs)

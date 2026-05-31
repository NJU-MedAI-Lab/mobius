from model.IMIS.segment_anything import sam_model_registry
from model.IMIS.model import IMISNet
from multiprocessing import Manager
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings("ignore")


def build_imis_model(args):
    model_type = args.get('imis_model_type', 'vit_b')
    test_mode = args.get('imis_test_mode', False)
    mask_num = args.get('imis_mask_num', 2)
    image_size = args.get('imis_image_size', 1024)
    sam_checkpoint = args.get('imis_sam_checkpoint',
                              '/workspace/EviMed/huggingface/IMISNet-B.pth')
    tokenizer = args.get('tokenizer', None)

    category_weights = 'model/IMIS/dataloaders/categories_weight.pkl'

    sam = sam_model_registry[model_type](image_size, sam_checkpoint)
    print("Loaded SAM model:", sam_checkpoint)
    imis = IMISNet(sam, tokenizer=tokenizer, test_mode=test_mode,
                   select_mask_num=mask_num, category_weights=category_weights)
    return imis

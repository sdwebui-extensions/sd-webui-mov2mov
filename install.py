import os
import hashlib
import launch
import modules.paths as ph
from modules import cmd_args

parser = cmd_args.parser
cmd_opts, _ = parser.parse_known_args()

# Install requirements if not installed
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(f"install {lib}", f"requirement for mov2mov: {lib}")

# Download ModNet models if not present
modnet_models_path = cmd_opts.data_dir + '/models/mov2mov-ModNet'

modnet_photographic_path = f"{modnet_models_path}/modnet_photographic_portrait_matting.ckpt"
modnet_webcam_path = f"{modnet_models_path}/modnet_webcam_portrait_matting.ckpt"

modnet_photographic_url = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt"
modnet_photographic_checksum = "db1f7ec96b370abebbd506e360ce9819380cb45bb99930f1e955dfcbe9e4035708a6190ebf292e0b8b740cefdc879bad9146ba8158baac0a702aadf5311a8cd6"

modnet_webcam_url = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_webcam_portrait_matting.ckpt"
modnet_webcam_checksum = "19ec6baa9934f834739d496c6f5d119a4e6fe09f67d19342025d3e9199b694814bc89d4e866db1d11da5e1740eba72a2ff4116420c5c212bf886d330d526f603"

# Compare checksum to make sure downloaded file is not modified
def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename,'rb') as f: 
        while chunk := f.read(chunk_num_blocks*h.block_size): 
            h.update(chunk)
    return h.hexdigest()

# Download modnet_photographic_portrait_matting.ckpt
if not os.path.exists(modnet_photographic_path):
    print('Downloading model for mov2mov ModNet, this is a one time operation')
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url(modnet_photographic_url, model_dir=modnet_models_path)

    if checksum(modnet_photographic_path) != modnet_photographic_checksum:
        os.remove(modnet_photographic_path)
        print(f"\nWarning: unable to automatically downloading ModNet model for mov2mov (checksum mismatch), please manually download from {modnet_photographic_url} and place in the folder {modnet_models_path}\n")
    else:
        print('Model download for mov2mov ModNet completed\n')

# Download modnet_webcam_portrait_matting.ckpt
if not os.path.exists(modnet_webcam_path):
    print('Downloading model for mov2mov ModNet, this is a one time operation')
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url(modnet_webcam_url, model_dir=modnet_models_path)

    if checksum(modnet_webcam_path) != modnet_webcam_checksum:
        os.remove(modnet_webcam_path)
        print(f"\nWarning: unable to automatically downloading ModNet model for mov2mov (checksum mismatch), please manually download from {modnet_webcam_url} and place in the folder {modnet_models_path}\n")
    else:
        print('Model download for mov2mov ModNet completed\n')

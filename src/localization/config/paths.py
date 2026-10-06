from dataclasses import dataclass
from pathlib import Path
import os   
#ROOT_DIR = Path(os.getenv("ROOT_DIR", Path(__file__).resolve().parent.parent.parent))
ROOT_DIR =  Path("/var/tmp/localization")


#ROOT_DIR = Path(os.getenv("ROOT_DIR", None))

#ROOT_DIR = Path("/var/tmp/localization")

#ROOT_DIR = Path(__file__).resolve().parent.parent.parent

@dataclass
class PathConfig:
    root_dir:Path = ROOT_DIR
    
    data_dir:Path = root_dir / "data"
    dichasus:Path = data_dir / "dichasus"
    passive_train:Path = data_dir / "passive/train.h5"
    passive_p1_test:Path = data_dir / "passive/test.h5"
    passive_p2_test:Path = data_dir / "passive/test2.h5"
    fiveg_dir:Path = data_dir / "5G"
    fiveg_train:Path= fiveg_dir / "training_data"
    fiveg_test:Path= fiveg_dir / "test_data"
    pt_dir:Path = data_dir/"pt" 
    
    #csv_dir:Path = ROOT_DIR / "csv"
    lightning_logs:Path = root_dir / "lightning_logs/"
    reports:Path = root_dir / "reports"
    
    
    map_5g : Path = data_dir / "5G/5g_ground_truth.png"
    map_dichasus:Path = data_dir / "dichasus/dichasus_ground_truth.png"
    
    @property
    def dichasus_list(self)->list:
        return [self.dichasus / x for x in[ "dichasus-cf02.tfrecords","dichasus-cf03.tfrecords","dichasus-cf04.tfrecords"] ]


if __name__=="__main__":
    path = PathConfig()
    x = path.pt_dir/"csi.npy"
    print(x)
    print(x.exists())

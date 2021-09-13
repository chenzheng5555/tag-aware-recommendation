from utility.word import CFG, Global
from utility.utils import printc, init_seed
from com import model_dict

import torch
import warnings
import multiprocessing

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    GLO = Global()   # 多线程只会多次执行外面
    #GLO.pool = multiprocessing.Pool(CFG['cpu_core'])

    init_seed(CFG['seed'])
    model_name = CFG['model']
    if CFG['model'][:4] == 'dtag':
        model_name = 'dtag'
    model, train, test = model_dict[model_name](GLO)
    printc(f"config:{CFG}")
    train.run(model)

    #model = torch.load(f"{GLO.out_dir}/model.pth.tar") #有时会出错
    model.load_state_dict(torch.load(f"{GLO.out_dir}/model.pth.tar", map_location=CFG['device']))

    if CFG['has_val']:
        results = test.run(model, istest=True)
        printc(f"test result: [{results}]")
        if GLO.writer:
            GLO.writer.add_text("LOG", f"test results: [{results}]")

    results = test.run(model, istest=True, group_k=4)
    printc(f"group: [{results}]")

    if GLO.writer:
        GLO.writer.add_text("LOG", f"group results: [{results}]")
        GLO.writer.add_text("LOG", f"config: [{CFG}], out_path={GLO.out_dir}")
        GLO.writer.close()
    
    if GLO.pool:
        GLO.pool.close()

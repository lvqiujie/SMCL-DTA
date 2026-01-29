import torch
from torch_geometric.data import InMemoryDataset

class GNNDataset(InMemoryDataset):
    def __init__(self, root, types='train', transform=None, pre_transform=None, pre_filter=None,
                 use_surface=False, use_masif=False):
        """
        初始化GNNDataset

        Args:
            root: 数据根目录
            types: 数据类型，'train', 'test1' 或 'test2'
            transform: 数据变换
            pre_transform: 预处理变换
            pre_filter: 过滤函数
            use_surface: 是否使用分子表面特征
            use_masif: 是否使用蛋白质MaSIF表面特征
            download_pdb: 是否自动下载缺失的PDB文件
            n_jobs: 并行作业数（当前不使用，保留参数以保持兼容性）
            protein_map: 蛋白质映射文件路径（可选）
        """
        self.use_surface = use_surface  # 是否使用分子表面特征
        self.use_masif = use_masif  # 是否使用蛋白质MaSIF表面特征
        super().__init__(root, transform, pre_transform, pre_filter)

        if types == 'train':
            print(self.processed_paths)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            print("")
        elif types == 'test1':
            self.data, self.slices = torch.load(self.processed_paths[1], weights_only=False)
            print("")
        elif types == 'test2':
            self.data, self.slices = torch.load(self.processed_paths[2], weights_only=False)

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv', 'data_test2.csv']

    @property
    def processed_file_names(self):
        # 根据使用的特征生成文件名
        suffix = []
        if self.use_surface:
            suffix.append("surface")
        if self.use_masif:
            suffix.append("masif")

        suffix_str = "_".join(suffix)
        if suffix_str:
            suffix_str = "_" + suffix_str

        return [
            f'processed_data_train{suffix_str}.pt',
            f'processed_data_test1{suffix_str}.pt',
            f'processed_data_test2{suffix_str}.pt'
        ]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process(self):
        pass

if __name__ == "__main__":
    dataset = GNNDataset('data/davis', types='train', use_surface=True, use_masif=True)
    dataset = GNNDataset('data/davis', types='test1', use_surface=True, use_masif=True)

    print(dataset)
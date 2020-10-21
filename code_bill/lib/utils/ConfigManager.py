import os

from ..settings.config import settings


class Config(object):
    def __init__(self):
        self.settings = settings

    def get_property(self, property_name):
        return self.settings[property_name]


class PathConfig(Config):
    def get_property(self, property_name):
        """
        override
        """
        return self.settings.path[property_name]

    def _create_dir_if_not(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def _get_path(self, pathname):
        path = self.get_property(pathname)
        self._create_dir_if_not(path)
        return path

    def _check_path(self, pathname):
        path = self.get_property(pathname)
        if not os.path.isdir(path):
            raise f"[PathConfig] {pathname}: {path} doesn't exist!"
        return path

    @property
    def dir_data(self):
        return self._check_path('dir_data')

    @property
    def dir_checkpoint(self):
        return self._get_path('dir_checkpoint')

    @property
    def dir_figures(self):
        return self._get_path('dir_figures')

    @property
    def dir_bestmodel(self):
        return self._get_path('dir_bestmodel')

    @property
    def dir_logging(self):
        return self._get_path('dir_logging')

    @property
    def dir_results(self):
        return self._get_path('dir_results')

    @property
    def dir_features(self):
        return self._get_path('dir_features')

    @property
    def dir_advimages(self):
        return self._get_path('dir_advimages')

    @property
    def fn_attack_resume(self):
        return self.get_property('fn_attack_resume')

    @property
    def fn_train_resume(self):
        return self.get_property('fn_train_resume')


def test():
    dir_data = "../PlantVillage-Dataset/raw/color/"
    dir_checkpoint = "./checkpoint/"
    dir_bestmodel = "./bestmodel/"
    dir_logging = "./logging/"
    dir_results = "./results/"
    dir_figures = "./figures/"
    dir_features = "./features/"
    dir_advimages = "./advimages/"
    fn_attack_resume = "attack_resume.ini"
    fn_train_resume = "resume.ini"

    pc = PathConfig()
    assert(dir_data == pc.dir_data)
    assert(dir_checkpoint == pc.dir_checkpoint)
    assert(dir_bestmodel == pc.dir_bestmodel)
    assert(dir_logging == pc.dir_logging)
    assert(dir_results == pc.dir_results)
    assert(dir_figures == pc.dir_figures)
    assert(dir_features == pc.dir_features)
    assert(dir_advimages == pc.dir_advimages)
    assert(fn_attack_resume == pc.fn_attack_resume)
    assert(fn_train_resume == pc.fn_train_resume)

    c = Config()
    print(settings.feature)


if __name__ == '__main__':
    test()

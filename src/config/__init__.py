import yaml 
import os

class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

def get_config(name):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, name + ".yaml"), 'r') as stream:
        cfg = yaml.load(stream, Loader=Loader)
    return cfg
import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


# load_sd是否加载预训练的模型
def make(model_spec, state_dict=None, args=None, load_sd=False):
    if args is not None:  # model_spec是一个关于模型的字典 有名称 也有可能有args
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)  # 前面models是一个字典 键是模型名称 值是模型对应的类 这里相当于根据args调了model构造函数
    if load_sd:
        # model.load_state_dict(model_spec['sd'])
        model.load_state_dict(state_dict)

    return model

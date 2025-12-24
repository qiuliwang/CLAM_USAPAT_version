from models.resnet_custom import resnet50_baseline


model = resnet50_baseline(pretrained=True)
print(model)
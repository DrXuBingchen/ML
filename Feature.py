from radiomics import featureextractor
import os
import pandas as pd
import SimpleITK as sitk
df = pd.DataFrame()
settings = {}
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = [1, 1, 1]  # unit: mm
settings['interpolator'] = sitk.sitkNearestNeighbor
settings['normalize'] = True
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableImageTypeByName('LoG')
extractor.enableImageTypeByName('Wavelet')
extractor.enableAllFeatures()
for folder in folders:
    files = os.listdir(os.path.join(basePath, folder))
    for file in files:
        if file.endswith('image.nrrd'):
            imageFile = os.path.join(basePath, folder, file)
        if file.endswith('mask.nrrd'):
            maskFile = os.path.join(basePath, folder, file)
    featureVector = extractor.execute(imageFile, maskFile)
    df_new = pd.DataFrame.from_dict(featureVector.values()).T
    df_new.columns = featureVector.keys()
    df_new.insert(0, 'imageFile', imageFile)
    df = pd.concat([df, df_new])

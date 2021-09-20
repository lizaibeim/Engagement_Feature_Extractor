from overlap_detector import OverlapDetector

if __name__ == '__main__':
    # # ZCR
    labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr"
    path_to_store = "D:\SpectrumAnalysis\\resBlstm_zcr_3classes_unaugmented_weighted_60epochs"
    osd = OverlapDetector(images_dir, labels_path, imagetype='zcr')
    # osd.train_model(path_to_store, epochs=60, batch_size=16, augmented=False, weighted=True)
    osd.populate_model("D:\SpectrumAnalysis\\ckt", augmented=False, weighted=True, images_dir=images_dir, labels_path=labels_path)
    osd.evaluation('val')
    osd.evaluation('test')
    osd.populate_model("D:\SpectrumAnalysis\\resBlstm_zcr_3classes_unaugmented_weighted_60epochs", augmented=False, weighted=True, images_dir=images_dir,
                       labels_path=labels_path)
    osd.evaluation('val')
    osd.evaluation('test')
    osd.populate_model("D:\SpectrumAnalysis\\models\\reslstm_zcr_3classes_unaugmented_weighted_60epochs", augmented=False, weighted=True, images_dir=images_dir,
                       labels_path=labels_path)
    osd.evaluation('val')
    osd.evaluation('test')


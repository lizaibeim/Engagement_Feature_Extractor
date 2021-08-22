from overlap_detector import OverlapDetector

if __name__ == '__main__':
    # Gray
    # labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    # images_dir = "D:\SpectrumAnalysis\mel_spectrum_gray"
    # od = OverlapDetector(images_dir, labels_path, imagetype='gray')
    # p1 = 'D:\SpectrumAnalysis\models\cnnlstm_gray_3classes_unaugmented_unweighted_100epochs'
    # od.train_model(p1, 100, 16, False, False)
    # od.evaluation_on_val()
    #
    # p2 = 'D:\SpectrumAnalysis\models\cnnlstm_gray_3classes_unaugmented_weighted_100epochs'
    # od.train_model(p2, 100, 16, True, False)
    # od.evaluation_on_val()
    #
    # p3 = 'D:\SpectrumAnalysis\models\cnnlstm_gray_3classes_augmented_unweighted_100epochs'
    # od.train_model(p3, 100, 16, False, True)
    # od.evaluation_on_val()
    #
    # p4 = 'D:\SpectrumAnalysis\models\cnnlstm_gray_3classes_augmented_weighted_100epochs'
    # od.train_model(p4, 100, 16, True, True)
    # od.evaluation_on_val()
    #
    # # Viridis
    # labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    # images_dir = "D:\SpectrumAnalysis\mel_spectrum_viridis"
    # od2 = OverlapDetector(images_dir, labels_path, imagetype='rgb')
    #
    # p5 = 'D:\SpectrumAnalysis\models\cnnlstm_viridis_3classes_unaugmented_unweighted_100epochs'
    # od2.train_model(p5, 100, 16, False, False)
    # od2.evaluation_on_val()
    #
    # p6 = 'D:\SpectrumAnalysis\models\cnnlstm_viridis_3classes_unaugmented_weighted_100epochs'
    # od2.train_model(p6, 100, 16, True, False)
    # od2.evaluation_on_val()
    #
    # p7 = 'D:\SpectrumAnalysis\models\cnnlstm_viridis_3classes_augmented_unweighted_100epochs'
    # od2.train_model(p7, 100, 16, False, True)
    # od2.evaluation_on_val()
    #
    # p8 = 'D:\SpectrumAnalysis\models\cnnlstm_viridis_3classes_augmented_weighted_100epochs'
    # od2.train_model(p8, 100, 16, True, True)
    # od2.evaluation_on_val()
    #
    # # ZCR
    labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr"
    # od3 = OverlapDetector(images_dir, labels_path, imagetype='rgb')
    #
    # p9 = 'D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_unaugmented_unweighted_100epochs'
    # od3.train_model(p9, 200, 16, False, False)
    # od3.evaluation_on_val()
    #
    # p10 = 'D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_unaugmented_weighted_100epochs'
    # od3.train_model(p10, 200, 16, True, False)
    # od3.evaluation_on_val()
    #
    # p11 = 'D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_augmented_unweighted_100epochs'
    # od3.train_model(p11, 200, 16, False, True)
    # od3.evaluation_on_val()

    p12 = 'D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_augmented_weighted_300epochs'
    od3 = OverlapDetector(images_dir, labels_path, imagetype='rgb')
    # od3.populate_model(p12, True, True)
    od3.train_model(p12, 300, 16, True, True)
    od3.evaluation_on_val()

    # labels_path = "D:\SpectrumAnalysis\labelling_results\OverlapLabels.xlsx"
    # images_dir = "D:\SpectrumAnalysis\mel_spectrum_zcr"
    # p12 = 'D:\SpectrumAnalysis\models\cnnlstm_zcr_3classes_augmented_weighted_100epochs'
    # od = OverlapDetector(images_dir, labels_path, imagetype='rgb')
    # od.train_model(p12, 200, 16, True, True)
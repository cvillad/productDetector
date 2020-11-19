package com.example.productDetector;

import java.io.IOException;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;

import static org.pytorch.torchvision.TensorImageUtils.TORCHVISION_NORM_MEAN_RGB;
import static org.pytorch.torchvision.TensorImageUtils.TORCHVISION_NORM_STD_RGB;

final public class TiendaClassificationModel {
    private TiendaClassificationModel(){
    }

    public static ZooModel<Image, Classifications> loadModel() throws ModelException, IOException{
        Pipeline pipeline = new Pipeline();

        pipeline.add(new Resize(240, 240, Image.Interpolation.BILINEAR))
                .add(new ToTensor())
                .add(new Normalize(TORCHVISION_NORM_MEAN_RGB,
                        TORCHVISION_NORM_STD_RGB));

        ImageClassificationTranslator translator = ImageClassificationTranslator.builder()
                .setPipeline(pipeline)
                .optApplySoftmax(true)
                .optSynset(TiendaClassesNames.TIENDA_CLASSES)
                .build();

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelUrls("https://sagemaker-us-west-2-256305374409.s3-us-west-2.amazonaws.com/pytorch/fastai.zip")
                .optTranslator(translator)
                .build();

        return ModelZoo.loadModel(criteria);
    }

    public static void predictCropImages(NDList detections, NDList imgsNDList,
                                         Predictor<Image, Classifications> tiendaModelPredictor)
                                                                            throws TranslateException {
        int i = 0;
        for (NDArray d: detections){
            NDArray temp = d.duplicate();
            System.out.println(imgsNDList.get(0).getShape().slice(0,2));
            System.out.println(imgsNDList.get(1).getShape());

            temp = TiendaLocalizationModel.scaleCoords(imgsNDList.get(0).getShape().slice(0,2),
                    temp.get(":,:4"), imgsNDList.get(1).getShape());
            int n = Math.toIntExact(temp.getShape().get(0));
            NDArray probs = TiendaLocalizationModel.manager.zeros(new Shape(n), temp.getDataType());
            NDArray realClasses = TiendaLocalizationModel.manager.zeros(new Shape(n), temp.getDataType());

            for(int j=0; j<temp.getShape().get(0); j++){
                StringBuilder sb = new StringBuilder();
                NDArray a = temp.get(String.valueOf(j)+", :");
                sb.append((int) a.get(1).getFloat()).append(":")
                        .append((int) a.get(3).getFloat()).append(",")
                        .append((int) a.get(0).getFloat()).append(":")
                        .append((int) a.get(2).getFloat());

                System.out.println(sb);
                NDArray im = imgsNDList.get(1).get(new NDIndex(sb.toString()));
                //im = TiendaLocalizationModel.imgArrayRGBtoBGRandBGRtoRGB(im);

                Image imageCrop = ImageFactory.getInstance().fromNDArray(im.transpose(2, 0, 1));
                Classifications classification = tiendaModelPredictor.predict(imageCrop);

                probs.set(new NDIndex(String.valueOf(j)), classification.best().getProbability());
                realClasses.set(new NDIndex(String.valueOf(j)),
                        TiendaClassesNames.TIENDA_CLASSES.indexOf(classification.best().getClassName()));
                System.out.println(classification.best().getClassName());
            }
            detections.get(i).set(new NDIndex(":,5"), realClasses);
            detections.get(i).set(new NDIndex(":,4"), probs);
            i++;
        }
        //return detections;
    }

}

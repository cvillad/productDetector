package com.example.productDetector;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.translator.YoloTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Pipeline;

final public class TiendaLocalizationModel {

    final static NDManager manager = NDManager.newBaseManager();
    public static Block block;
    public static ParameterStore parameterStore;
    public static NDArray anchorGrid,  stride, anchors;
    public static int no;
    public static DetectedObjects yoloObjectsDetected;
    public static int[][] colors;
    private TiendaLocalizationModel(){

    }

    public static void setYoloConfigurations(Map<String, Object> modelConfig, Model yoloTiendaModel){
        block = yoloTiendaModel.getBlock();
        parameterStore = new ParameterStore(manager, true);
        ArrayList<ArrayList<Integer>> anchorsList =  (ArrayList<ArrayList<Integer>>) modelConfig.get("anchors");
        int nl = anchorsList.size();
        int nc = 21;
        no = nc + 5;

        int[][] anchorsArray = new int[nl][anchorsList.get(0).size()];

        int i = 0;
        for (ArrayList<Integer> inList : anchorsList){
            int j = 0;
            for (Integer num: inList){
                anchorsArray[i][j++] = num.intValue();
            }
            i++;
        }

        NDArray a = manager.create(anchorsArray).toType(DataType.FLOAT32, false).reshape(nl, -1, 2);
        anchors = a.duplicate();
        anchorGrid = a.duplicate().reshape(nl, 1, -1, 1, 1, 2);

        NDList tempForwardInput = new NDList();
        tempForwardInput.add(manager.zeros(new Shape(1,3,128,128)));
        NDList tempForward = block.forward(parameterStore, tempForwardInput, false);
        stride = manager.zeros(new Shape(tempForward.size()), DataType.FLOAT32);

        i = 0;
        for(NDArray x: tempForward){
            stride.set(new NDIndex(i++), 128.0/x.getShape().get(x.getShape().dimension() - 2));
        }
        anchors = anchors.div(stride.reshape(-1, 1, 1));
        colors = new int[TiendaClassesNames.YOLO_CLASS.size()][3];
        i = 0;

        Random rand = new Random();
        for(String s: TiendaClassesNames.YOLO_CLASS){
            for(int j=0;j<3;j++){
                colors[i][j] = rand.nextInt(256);
            }
            i++;
        }
    }

    public static ZooModel<Image, DetectedObjects> loadModel() throws ModelException, IOException {
        Pipeline pipeline = new Pipeline();
        pipeline.add(ndArray -> ndArray.toType(DataType.FLOAT32, false))
                .add(ndArray -> ndArray.transpose(2, 0 , 1).div(255.0))
                .add(ndArray -> ndArray.expandDims(0));

        YoloTranslator translator = YoloTranslator.builder()
                .setPipeline(pipeline)
                .optSynset(TiendaClassesNames.YOLO_CLASS)
                .build();

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .setTypes(Image.class, DetectedObjects.class)
                .optModelUrls("https://sagemaker-us-west-2-256305374409.s3-us-west-2.amazonaws.com/pytorch/yolov5_s.zip")
                .optTranslator(translator)
                .build();

        return ModelZoo.loadModel(criteria);
    }

    public static NDArray imgArrayRGBtoBGRandBGRtoRGB(NDArray imgArrayRGB){
        NDArray newImgArrayBGR = manager.zeros(imgArrayRGB.getShape(), DataType.UINT8);
        char[] index = ":,:,0".toCharArray();
        char[] indexTemp = ":,:,0".toCharArray();
        for (int i = 2; i>=0; i--){
            index[index.length - 1] = String.valueOf(i).toCharArray()[0];
            indexTemp[indexTemp.length - 1] = String.valueOf(2-i).toCharArray()[0];
            newImgArrayBGR.set(new NDIndex(new String(index)), imgArrayRGB.get(new String(indexTemp)));
        }
        return newImgArrayBGR;
    }

    public static NDList letterBox(NDArray imgArrayRGB, Shape newShape, boolean auto,
                                   boolean scaleFill, boolean scaleUp){

        NDArray ndArray = imgArrayRGBtoBGRandBGRtoRGB(imgArrayRGB);
        Shape shape = ndArray.getShape().slice(0, 2);

        double r = Math.min((double) newShape.get(0)/shape.get(0),
                (double) newShape.get(1)/shape.get(1));
        if (!scaleUp){
            r = Math.min(r, 1.0);
        }
        double[] ratio = {r, r};
        long[] newUnpad = {Math.round(shape.get(1) * r),
                Math.round(shape.get(0) * r)};
        double dw = newShape.get(1) - newUnpad[0], dh = newShape.get(0) - newUnpad[1];
        if (auto){
            dw %= 32;
            dh %= 32;
        }else if (scaleFill){
            dw = 0.0;
            dh = 0.0;
            newUnpad[0] = newShape.get(1);
            newUnpad[1] = newShape.get(0);
            ratio[0] = (double) newShape.get(1)/shape.get(1);
            ratio[1] = (double) newShape.get(0)/shape.get(0);
        }

        dw /= 2;
        dh /= 2;

        NDArray tempShape = manager.create(new long[] {shape.get(1),
                shape.get(0)}, new Shape(2));
        NDArray tempNewUnpad = manager.create(newUnpad, new Shape(2));

        int rows = (int) ndArray.getShape().get(0);
        int cols = (int) ndArray.getShape().get(1);

        Mat src = new Mat(rows, cols, CvType.CV_8UC3);
        Mat dst = new Mat();
        byte[] data;

        if (tempShape.neq(tempNewUnpad).all().getBoolean()){
            Size size = new Size((double)  newUnpad[0], (double)  newUnpad[1]);
            src.put(0, 0, ndArray.toByteArray());

            Imgproc.resize(src, dst, size, 0, 0, Imgproc.INTER_LINEAR);
            data = new byte[(int) dst.total()*dst.channels()];
            dst.get(0, 0, data);
            ndArray = manager.create(data, new Shape(dst.rows(), dst.cols(), dst.channels()));
        }


        int top = (int) Math.round(dh - 0.1);
        int bottom = (int) Math.round(dh + 0.1);
        int left = (int) Math.round(dw - 0.1);
        int right = (int) Math.round(dw + 0.1);


        rows = (int) ndArray.getShape().get(0);
        cols = (int) ndArray.getShape().get(1);

        src = new Mat(rows, cols, CvType.CV_8UC3);
        src.put(0, 0, ndArray.toByteArray());
        dst = new Mat();

        Scalar color = new Scalar(new double[] {114,114,114});
        Core.copyMakeBorder(src, dst, top, bottom, left, right, Core.BORDER_CONSTANT, color);

        data = new byte[(int) dst.total()*dst.channels()];
        dst.get(0, 0, data);
        ndArray = manager.create(data, new Shape(dst.rows(), dst.cols(), dst.channels()));


        ndArray = imgArrayRGBtoBGRandBGRtoRGB(ndArray);

        NDList resultNDArray = new NDList();
        resultNDArray.add(ndArray);
        resultNDArray.add(imgArrayRGB);

        return resultNDArray;
    }

    public static NDList predict(NDArray ndArrayInput){
        NDList yoloOutput = null;
        // Define pipeline
        Pipeline pipeline = new Pipeline();
        pipeline.add(ndArray -> ndArray.toType(DataType.FLOAT32, false))
                .add(ndArray -> ndArray.transpose(2, 0 , 1).div(255.0))
                .add(ndArray -> ndArray.expandDims(0));
        //create Batch
        NDList batch = new NDList();
        batch.add(ndArrayInput);
        //preprocess batch
        NDList preprocessBatch = pipeline.transform(batch);
        //block
        yoloOutput = block.forward(parameterStore, preprocessBatch, false);
        return yoloOutput;
    }

    public static NDList meshgrid(int nx, int ny){
        NDList resultMeshgrid = new NDList();
        NDArray nxArray = manager.arange(nx);
        NDArray nyArray = manager.arange(ny);
        NDArray nxMeshgrid = nxArray.tile(ny).reshape(-1, nx);
        NDArray nyMeshgrid = nyArray.tile(nx).reshape(-1, ny).transpose();
        resultMeshgrid.add(nyMeshgrid);
        resultMeshgrid.add(nxMeshgrid);
        return resultMeshgrid;
    }

    public static NDArray makeGrid(int nx, int ny){
        NDList listMeshgrid = meshgrid(nx, ny);
        NDArray yv = listMeshgrid.get(0);
        NDArray xv = listMeshgrid.get(1);
        return xv.stack(yv, 2).reshape(1, 1, ny, nx, 2).toType(DataType.FLOAT32, false);
    }

    public static NDArray preprocessYoloOutput(NDList yoloOutput){
        NDList z = new NDList();
        int i = 0;
        for(NDArray x: yoloOutput){
            int bs =(int) x.getShape().get(0),
                    ny =(int) x.getShape().get(2),  nx = (int) x.getShape().get(3);
            NDArray grid = makeGrid(nx, ny);
            NDArray y = x.getNDArrayInternal().sigmoid().duplicate();
            NDArray temp = y.get("..., 0:2").duplicate();

            y.set(new NDIndex("..., 0:2"), (temp.mul(2.0).sub(0.5).add(grid)).mul(stride.get(i)));

            temp = y.get("..., 2:4").duplicate();
            y.set(new NDIndex("..., 2:4"), (temp.mul(2)).pow(2).mul(anchorGrid.get(i)));

            z.add(y.reshape(bs, -1, no));
            i++;
        }
        return NDArrays.concat(z, 1);
    }

    public static NDArray xywh2xyxy(NDArray x){
        NDArray y = manager.zeros(x.getShape(), x.getDataType());
        y.set(new NDIndex(":, 0"), x.get(":, 0").sub(x.get(":, 2").div(2)));
        y.set(new NDIndex(":, 1"), x.get(":, 1").sub(x.get(":, 3").div(2)));
        y.set(new NDIndex(":, 2"), x.get(":, 0").add(x.get(":, 2").div(2)) );
        y.set(new NDIndex(":, 3"), x.get(":, 1").add(x.get(":, 3").div(2)));
        return y;
    }

    public static NDArray xyxy2xywh(NDArray x){
        NDArray y = manager.zeros(x.getShape(), x.getDataType());
        y.set(new NDIndex(":, 0"), (x.get(":, 0").add(x.get(":, 2")).div(2)));
        y.set(new NDIndex(":, 1"), (x.get(":, 1").add(x.get(":, 3")).div(2)));
        y.set(new NDIndex(":, 2"), x.get(":, 2").sub(x.get(":, 0")));
        y.set(new NDIndex(":, 3"), x.get(":, 3").sub(x.get(":, 1")));
        return y;
    }

    public static NDArray clipCoords(NDArray x, Shape imgShape){
        NDArray y = manager.zeros(x.getShape(), x.getDataType());
        y.set(new NDIndex(":, 0"), x.get(":,0").clip(0, imgShape.get(1)));
        y.set(new NDIndex(":, 1"), x.get(":,1").clip(0, imgShape.get(0)));
        y.set(new NDIndex(":, 2"), x.get(":,2").clip(0, imgShape.get(1)) );
        y.set(new NDIndex(":, 3"), x.get(":,3").clip(0, imgShape.get(0)) );
        return y;
    }

    public static NDArray scaleCoords(Shape img1Shape, NDArray coords, Shape img0Shape){
        double gain = Math.min((double) img1Shape.get(0)/img0Shape.get(0),
                (double) img1Shape.get(1)/img0Shape.get(1)); //gain = old/new
        double[] pad = new double[]{(double) (img1Shape.get(1)-img0Shape.get(1)*gain)/2,
                (double) (img1Shape.get(0)-img0Shape.get(0)*gain)/2};
        coords.set(new NDIndex(":, 0"), coords.get(":, 0").sub(pad[0]));
        coords.set(new NDIndex(":, 2"), coords.get(":, 2").sub(pad[0]));
        coords.set(new NDIndex(":, 1"), coords.get(":, 1").sub(pad[1]));
        coords.set(new NDIndex(":, 3"), coords.get(":, 3").sub(pad[1]));
        coords.set(new NDIndex(":, :4"), coords.get(":, :4").div(gain));
        return clipCoords(coords, img0Shape);
    }

    public static long[] reverse(long a[]){
        int i, k;
        long t;
        int n = a.length;
        for (i = 0;i < a.length/2; i++){
            t = a[i];
            a[i] = a[n-i-1];
            a[n-i-1] = t;
        }
        return a;
    }

    public static NDArray nms(NDArray dets, NDArray scores, double thresh){
        NDArray x1 = dets.get(":, 0");
        NDArray y1 = dets.get(":, 1");
        NDArray x2 = dets.get(":, 2");
        NDArray y2 = dets.get(":, 3");

        NDArray areas = (x2.sub(x1).add(1)).mul(y2.sub(y1).add(1));
        NDArray order = scores.argSort(0, true);
        order = manager.create(reverse(order.toLongArray()));

        ArrayList<Long> keep = new ArrayList<Long>();

        int n = (int) order.size();
        while(n > 1){
            NDArray i = order.get("0");
            keep.add(new Long(i.getLong()));

            NDArray xx1 = NDArrays.maximum(x1.get(String.valueOf(i.getLong())),
                    x1.get(new NDIndex().addPickDim(order.get("1:"))));
            NDArray yy1 = NDArrays.maximum(y1.get(String.valueOf(i.getLong())),
                    y1.get(new NDIndex().addPickDim(order.get("1:"))));
            NDArray xx2 = NDArrays.minimum(x2.get(String.valueOf(i.getLong())),
                    x2.get(new NDIndex().addPickDim(order.get("1:"))));
            NDArray yy2 = NDArrays.minimum(y2.get(String.valueOf(i.getLong())),
                    y2.get(new NDIndex().addPickDim(order.get("1:"))));
            NDArray w = NDArrays.maximum((float) 0.0, xx2.sub(xx1).add(1));
            NDArray h = NDArrays.maximum((float) 0.0, yy2.sub(yy1).add(1));
            NDArray inter = w.mul(h);
            NDArray ovr = inter.div(areas.get(String.valueOf(i.getLong()))
                    .add(areas.get(new NDIndex().addPickDim(order.get("1:"))))
                    .sub(inter));

            NDArray inds = NDArrays.where(ovr.lte(thresh), manager.arange(1, ovr.size() + 1),
                    manager.arange(1, ovr.size() + 1).mul(-1));

            inds = (inds.get(inds.gte(0))).sub(1);
            order = order.get(new NDIndex().addPickDim(inds.add(1)));
            n = (int) order.size();
        }

        NDArray keepArray = manager.zeros(new Shape(keep.size()));

        int i = 0;
        for(Long value: keep){
            keepArray.set(new NDIndex(""+String.valueOf(i)), value.intValue());
            i++;
        }
        return keepArray;
    }

    public static NDList NonMaxSuppression(NDArray pred, double confThresh, double iouThresh,
                                           boolean agnostic){
        int nc = (int) pred.get(0).getShape().get(1) - 5;
        NDArray xc = pred.get("..., 4").gt(confThresh);

        int minWh = 2, maxWh = 4096;
        int maxDet = 300;
        boolean redundant = true;
        boolean multiLabel = nc > 1;

        NDList output = new NDList();

        for(int i=0; i<pred.getShape().get(0); i++)
        {
            NDArray x = pred.get(i);
            NDList tempXList = new NDList();

            for (int j = 0; j < x.getShape().get(1); j++){
                tempXList.add(x.get(":,"+String.valueOf(j)).get(xc.get(i)).reshape(-1,1));
            }

            x = NDArrays.concat(tempXList, 1); //apply confidence

            if (x.getShape().get(0) == 0){
                continue;
            }

            x.set(new NDIndex(":, 5:"), x.get(":, 5:").mul(x.get(":, 4:5")));

            NDArray box = xywh2xyxy(x.get(":, :4"));
            if(multiLabel){
                NDList idxi = new NDList();
                NDList idxj = new NDList();
                NDList newX = new NDList();
                NDList subX = new NDList();

                NDArray confSw = x.get(":, 5:").gt(confThresh);
                for (i=0; i< confSw.getShape().get(0); i++){
                    NDArray tempLoopArray = confSw.get(String.valueOf(i));
                    NDArray colIndex = NDArrays.where(tempLoopArray.neq(0),
                            manager.arange(1, tempLoopArray.size() + 1),
                            manager.arange(1, tempLoopArray.size() + 1).mul(-1));
                    colIndex = (colIndex.get(colIndex.gte(0))).sub(1);
                    idxj.add(colIndex);
                    if (colIndex.size() > 0){
                        idxi.add(manager.create(new int[] {i}).repeat(colIndex.size()));
                    }else idxi.add(colIndex);
                }

                try{
                    NDArray idxiArray = NDArrays.concat(idxi, 0).toType(DataType.INT32, false);
                    NDArray idxjArray = NDArrays.concat(idxj, 0).toType(DataType.INT32, false);

                    newX.add(box.get(new NDIndex().addPickDim(idxiArray.repeat(
                            box.getShape().get(1))
                                    .reshape(-1, box.getShape().get(1)))
                            )
                    );
                    for (long h = 0; h<idxiArray.size(); h++){
                        String newIndex = String.valueOf(idxiArray.get(h).getInt()
                                +","+String.valueOf(idxjArray.get(h).add(5).getInt()));
                        subX.add(x.get(newIndex));
                    }
                    if (subX.size()!=0) {
                        newX.add(NDArrays.stack(subX).reshape(-1, 1));
                        newX.add(idxjArray.toType(DataType.FLOAT32, false).reshape(-1, 1));
                        x = NDArrays.concat(newX, 1);
                    }else x = manager.create(new Shape(0,2));

                }catch (Exception e){
                    x = manager.create(new Shape(0,2));
                }

            }else{
                NDArray conf = x.get(":, 5:").max(new int[] {1}, true);
                NDArray idxMax = x.get(":, 5:").argMax(1).reshape(conf.getShape());
                NDList tempList = new NDList();
                tempList.add(box);
                tempList.add(conf);
                tempList.add(idxMax.toType(DataType.FLOAT32, true));
                x = NDArrays.concat(tempList, 1);

                //apply confidence
                NDArray tempMask = conf.reshape(-1).gt(confThresh);

                tempXList = new NDList();

                for (int k = 0; k < x.getShape().get(1); k++){
                    tempXList.add(x.get(":,"+String.valueOf(k)).get(tempMask).reshape(-1,1));
                }
                x = NDArrays.concat(tempXList, 1);
            }

            int n = (int) x.getShape().get(0); //number of boxes
            if (n == 0){
                continue;
            }

            NDArray c = x.get(":, 5:6").mul((agnostic) ? 0:maxWh);
            NDArray boxes = x.get(":, :4").add(c);
            NDArray scores = x.get(":, 4");
            NDArray idxBoxes = nms(boxes, scores, iouThresh);

            if (idxBoxes.getShape().get(0) > maxDet){
                idxBoxes = idxBoxes.get(":"+String.valueOf(maxDet));
            }
            output.add(x.get(new NDIndex()
                    .addPickDim(idxBoxes.repeat(x.getShape()
                            .get(1)).reshape(-1, x.getShape().get(1)))));
        }
        return output;
    }

    public static Bitmap getImageWithBoxesOriginalSize(NDList detections, NDArray preprocessImageArray,
                                                      NDArray originalImageArray, Bitmap outputImage){

        //outputImage.eraseColor(Color.TRANSPARENT);
        NDArray boundingBoxes = detections.get(0).get(":, :4").duplicate();
        boundingBoxes = scaleCoords(preprocessImageArray.getShape(),
                boundingBoxes, originalImageArray.getShape()).round();

        int imageWidth = (int) originalImageArray.getShape().get(1);
        int imageHeight = (int) originalImageArray.getShape().get(0);
        int[] classIndices = detections.get(0).get(":, -1").toType(DataType.INT32, true).flatten().toIntArray();
        double[] probs = detections.get(0).get(":, -2").toType(DataType.FLOAT64, true).flatten().toDoubleArray();

        int detected = Math.toIntExact(detections.get(0).getShape().get(0));

        float[] boxLeft = boundingBoxes.get(":,0").toFloatArray();
        float[] boxTop = boundingBoxes.get(":,1").toFloatArray();
        float[] boxRight = boundingBoxes.get(":,2").toFloatArray();
        float[] boxBottom = boundingBoxes.get(":,3").toFloatArray();

        List<String> retClasses = new ArrayList<>(detected);
        List<Double> retProbs = new ArrayList<>(detected);
        List<BoundingBox> retBB = new ArrayList<>(detected);

        Canvas canvas = new Canvas(outputImage);

        // set the paint configure
        int stroke = 12;
        Paint paint = new Paint();
        paint.setColor(0xFF0000);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(stroke);
        paint.setAntiAlias(true);
        for (int i = 0; i < detected; i++) {
            retClasses.add(TiendaClassesNames.YOLO_CLASS.get(classIndices[i]));
            retProbs.add(probs[i]);
            Rectangle rect = new Rectangle(boxLeft[i], boxTop[i], boxRight[i], boxBottom[i]);
            retBB.add(rect);

            String className = TiendaClassesNames.YOLO_CLASS.get(classIndices[i]);
            int color = Color.rgb(colors[classIndices[i]][0],colors[classIndices[i]][1],
                    colors[classIndices[i]][2]);
            paint.setColor(color);
            canvas.drawRect(boxLeft[i], boxTop[i], boxRight[i], boxBottom[i], paint);
            drawText(canvas, color, className, (int) boxLeft[i],(int) boxTop[i], stroke, 12);
        }
        yoloObjectsDetected = new DetectedObjects(retClasses, retProbs, retBB);
        return outputImage;
    }

    public static void drawText(Canvas canvas, int color, String text, int x, int y, int stroke, int padding) {
        Paint paint = new Paint();
        Paint.FontMetrics metrics = paint.getFontMetrics();
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(color);
        paint.setAntiAlias(true);

        x += stroke / 2;
        y += stroke / 2;

        int width = (int) (paint.measureText(text) + padding * 2 - stroke / 2);
        // the height here includes ascent
        int height = (int) (metrics.descent - metrics.ascent);
        int ascent = (int) metrics.ascent;
        Rect bounds = new Rect(x, y, x + width+ 10, y + height +10);
        canvas.drawRect(bounds, paint);
        paint.setColor(Color.WHITE);
        // ascent in android is negative value, so y = y - ascent
        canvas.drawText(text, x + padding, y - ascent, paint);
    }
}

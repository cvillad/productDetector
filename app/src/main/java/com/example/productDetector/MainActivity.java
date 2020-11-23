package com.example.productDetector;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.YuvImage;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.size.Size;

import org.opencv.android.OpenCVLoader;
import org.yaml.snakeyaml.Yaml;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Map;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

public class MainActivity extends AppCompatActivity {
    private static final String AUTHORITY=
            BuildConfig.APPLICATION_ID+".provider";
    Yaml yaml = new Yaml();
    boolean swLoadModel;
    int MY_PERMISSIONS_WRITE_EXTERNAL_STORAGE = 0;
    byte[] data;
    Bitmap bmp;
    ImageView imageResult;
    View progressBar;
    TextView textView;
    Image imgTest;

    Map<String, Object> modelConfig;
    ZooModel<Image, Classifications> model;
    ZooModel<Image, DetectedObjects> yoloModel;
    Predictor<Image, Classifications> predictor;
    CameraView camera;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);
        progressBar = findViewById(R.id.loading_progress);
        textView = findViewById(R.id.resultText);
        camera = findViewById(R.id.camera);
        imageResult = findViewById(R.id.imageResult);

        if (OpenCVLoader.initDebug()) {
        }
        //request permission
        checkPermission();

        File dir = getFilesDir();
        System.setProperty("DJL_CACHE_DIR", dir.getAbsolutePath());

        try {
            modelConfig = (Map<String, Object>) yaml.load(getAssets().open("yolov5s.yaml"));
        } catch (IOException errorFile){
            Log.e("DemoTienda", "Error File", errorFile);
        }

        camera.setLifecycleOwner(this);
        NDManager manager = NDManager.newBaseManager();
        new UnpackTask().execute();

        camera.addFrameProcessor( frame -> {
            if (frame.getDataClass() == byte[].class) {
                data = frame.getData();
                Size s = frame.getSize();
                YuvImage yuv = new YuvImage(data, ImageFormat.NV21, s.getWidth(), s.getHeight(), null);
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                yuv.compressToJpeg(new Rect(0, 0, s.getWidth(), s.getHeight()), 100, stream);
                byte[] buf = stream.toByteArray();
                bmp = BitmapFactory.decodeByteArray(buf, 0, buf.length, null);
                Matrix matrix = new Matrix();
                matrix.postRotate(90);
                bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);
                Bitmap croppedBmp = scaleCenterCrop(bmp, camera.getHeight(), camera.getWidth());
                imgTest = ImageFactory.getInstance().fromImage(croppedBmp);
                getInference();
                //imageResult.setImageBitmap(new_bmp);
            }
        });
    }

    public void getInference(){
        NDManager manager = NDManager.newBaseManager();
        NDArray originalImageArray = imgTest.toNDArray(manager);
        NDList imgsNDList = TiendaLocalizationModel.letterBox(originalImageArray,
                new Shape(224, 224), true, false,
                true);
        NDArray preprocessImageArray = imgsNDList.get(0).duplicate();
        NDList outputYolo = TiendaLocalizationModel.predict(imgsNDList.get(0));
        NDArray outputYoloPreprocess = TiendaLocalizationModel.preprocessYoloOutput(outputYolo);
        NDList detections = TiendaLocalizationModel.NonMaxSuppression(outputYoloPreprocess,
                0.4, 0.6, true);

        /*try {
            TiendaClassificationModel.predictCropImages(detections, imgsNDList, predictor);
        } catch (TranslateException e) {
            e.printStackTrace();
        }*/

        StringBuilder sb = new StringBuilder();
        //Create Bitmap
        Bitmap bmp = Bitmap.createBitmap(this.bmp.getWidth(), this.bmp.getHeight(), Bitmap.Config.ARGB_8888);
        Bitmap outputImage = scaleCenterCrop(bmp, camera.getHeight(), camera.getWidth());
        if (detections.size() > 0){
            outputImage = TiendaLocalizationModel.getImageWithBoxesOriginalSize(detections,
                    preprocessImageArray, originalImageArray, outputImage);
            for (Classifications.Classification classification : TiendaLocalizationModel.yoloObjectsDetected.items()) {
                sb.append(classification.getClassName())
                        .append(": ")
                        .append(String.format("%.2f%%", 100 * classification.getProbability()))
                        .append("\n");
                System.out.println(classification);
            }
            System.out.println(sb);
        }else{
            sb.append("there is no detection");
            //outputImage = bmp;
        }
        imageResult.setImageBitmap(outputImage);
        textView.setText(sb.toString());
    }

    public Bitmap scaleCenterCrop(Bitmap source, int newHeight, int newWidth) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();

        // Compute the scaling factors to fit the new height and width, respectively.
        // To cover the final image, the final scaling will be the bigger
        // of these two.
        float xScale = (float) newWidth / sourceWidth;
        float yScale = (float) newHeight / sourceHeight;
        float scale = Math.max(xScale, yScale);

        // Now get the size of the source bitmap when scaled
        float scaledWidth = scale * sourceWidth;
        float scaledHeight = scale * sourceHeight;

        // Let's find out the upper left coordinates if the scaled bitmap
        // should be centered in the new size give by the parameters
        float left = (newWidth - scaledWidth) / 2;
        float top = (newHeight - scaledHeight) / 2;

        // The target rectangle for the new, scaled version of the source bitmap will now
        // be
        RectF targetRect = new RectF(left, top, left + scaledWidth, top + scaledHeight);

        // Finally, we create a new bitmap of the specified size and draw our new,
        // scaled bitmap onto it.
        Bitmap dest = Bitmap.createBitmap(newWidth, newHeight, source.getConfig());
        Canvas canvas = new Canvas(dest);
        canvas.drawBitmap(source, null, targetRect, null);

        return dest;
    }

    public void checkPermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED) {
            // Permission is not granted
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        MY_PERMISSIONS_WRITE_EXTERNAL_STORAGE);
            }
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
        if (yoloModel != null){
            yoloModel.close();
        }
        super.onDestroy();
    }

    @SuppressLint("StaticFieldLeak")
    public class UnpackTask extends AsyncTask<Void, Integer, Boolean> {
        @Override
        protected Boolean doInBackground(Void... params) {
            try {
                Log.i("DemoTienda", "loading tienda classification model");
                model = TiendaClassificationModel.loadModel();
                Log.i("DemoTienda", "loading tienda yolo model");
                yoloModel = TiendaLocalizationModel.loadModel();
                TiendaLocalizationModel.setYoloConfigurations(modelConfig, yoloModel);
                Log.i("DemoTienda", "loading models success");
                predictor = model.newPredictor();
                return true;
            } catch (IOException | ModelException e) {
                Log.e("DemoTienda", null, e);
            }
            return false;
        }

        @Override
        protected void onPostExecute(Boolean result){
            if (result){
                swLoadModel = true;
                progressBar.setVisibility(View.GONE);
                AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
                alertDialog.setTitle("Success");
                alertDialog.setMessage("Success to load model");
                alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                        (dialog, which) -> dialog.dismiss());
                alertDialog.show();
            }else{
                swLoadModel = false;
                AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
                alertDialog.setTitle("Error");
                alertDialog.setMessage("Failed to load model");
                alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                        (dialog, which) -> finish());
                alertDialog.show();
            }
        }
    }

}
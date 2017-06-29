package com.example.shpark.mnist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static android.R.id.edit;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/mnist.pb";
    private static final String MNIST_IMAGE_FILE = "t10k-images-idx3-ubyte";
    private static final String MNIST_LABEL_FILE = "t10k-labels-idx1-ubyte";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "predict_op";
    private static final int[] INPUT_SIZE = {28, 28};

    private TensorFlowInferenceInterface inferenceInterface = null;
    private ImageView imageView = null;
    private ImageView resizedImageView = null;
    private EditText editValue = null;
    private EditText labelValue = null;
    private Bitmap bitmap;
    private Bitmap resizeBitmap;
    private Canvas canvas;
    private Paint paint;
    private float startX = 0, startY = 0, endX = 0, endY = 0;
    private Path path;

    private MnistImageFile mnistImage;
    private MnistLabelFile mnistLabel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initMnistModel();

        imageView = (ImageView) this.findViewById(R.id.imageView);
        resizedImageView = (ImageView) this.findViewById(R.id.resizedImageView);
        editValue = (EditText)this.findViewById(R.id.editText);
        labelValue = (EditText)this.findViewById(R.id.labelText);

        try {
            mnistImage = new MnistImageFile(getAssetFilePath(MNIST_IMAGE_FILE), "r");
            mnistLabel = new MnistLabelFile(getAssetFilePath(MNIST_LABEL_FILE), "r");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (NullPointerException e) {
            e.printStackTrace();
        }


        int px = (int)TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 350, getResources().getDisplayMetrics());
        bitmap = Bitmap.createBitmap(px, px, Bitmap.Config.ARGB_8888);
        bitmap.eraseColor(Color.WHITE);
        canvas = new Canvas(bitmap);
        paint = new Paint();
        paint.setColor(Color.BLUE);
        paint.setStrokeWidth(80.f);
        paint.setAntiAlias(true);
        paint.setDither(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        imageView.setImageBitmap(bitmap);

        resizeBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        resizedImageView.setImageBitmap(resizeBitmap);

        imageView.setOnTouchListener(this);

        Button predictBtn = (Button)this.findViewById(R.id.predictBtn);
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Bitmap trimBitmap = trim(bitmap, Color.WHITE);

                resizeBitmap = Bitmap.createScaledBitmap(trimBitmap, 28, 28, true);

                int x = resizeBitmap.getWidth();
                int y = resizeBitmap.getHeight();
                int[] intArray = new int[x * y];
                resizeBitmap.getPixels(intArray, 0, 28, 0, 0, 28, 28);

                float[] floatArray = new float[intArray.length];
                for(int i=0; i<intArray.length; i++) {
                    floatArray[i] = ARGB8_to_RGB565(intArray[i]);
                }

                float[] keep_conv = {1.0f};
                float[] keep_hidden = {1.0f};
                int[] res = {0};
                inferenceInterface.feed(INPUT_NODE, floatArray, 1, 28, 28, 1);
                inferenceInterface.feed("keep_conv", keep_conv);
                inferenceInterface.feed("keep_hidden", keep_hidden);
                inferenceInterface.run(new String[] {OUTPUT_NODE});
                inferenceInterface.fetch(OUTPUT_NODE, res);

                editValue.setText(String.valueOf(res[0]));

                resizedImageView.setImageBitmap(resizeBitmap);
                resizedImageView.invalidate();
            }
        });

        Button clearBtn = (Button)this.findViewById(R.id.clearBtn);
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                bitmap.eraseColor(Color.WHITE);
                resizeBitmap.eraseColor(Color.WHITE);
                editValue.setText("");
                labelValue.setText("");
            }
        });

        Button testBtn = (Button)this.findViewById(R.id.testBtn);
        testBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    int[] data = mnistImage.readImage2();

                    float[] floatArray = new float[data.length];
                    for(int i=0; i<data.length; i++) {
                        floatArray[i] = data[i];
                    }

                    float[] keep_conv = {1.0f};
                    float[] keep_hidden = {1.0f};
                    int[] res = {0};
                    inferenceInterface.feed(INPUT_NODE, floatArray, 1, 28, 28, 1);
                    inferenceInterface.feed("keep_conv", keep_conv);
                    inferenceInterface.feed("keep_hidden", keep_hidden);
                    inferenceInterface.run(new String[] {OUTPUT_NODE});
                    inferenceInterface.fetch(OUTPUT_NODE, res);

                    for(int i=0; i<data.length; i++) {
                        data[i] = RGB_to_ARGB(data[i]);
                    }

                    Bitmap testBitmap  = Bitmap.createBitmap(data, 0, 28, 28, 28, Bitmap.Config.ARGB_8888);

                    resizeBitmap = testBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    resizedImageView.setImageBitmap(resizeBitmap);
                    resizedImageView.invalidate();

                    bitmap = Bitmap.createScaledBitmap(resizeBitmap, bitmap.getWidth(), bitmap.getHeight(), true);
                    canvas.setBitmap(bitmap);
                    imageView.setImageBitmap(bitmap);
                    imageView.invalidate();

                    editValue.setText(String.valueOf(res[0]));
                    labelValue.setText(String.valueOf(mnistLabel.readLabel()));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });

    }

    private void initMnistModel() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction();
        switch (action) {
            case MotionEvent.ACTION_DOWN:
                startX=event.getX();
                startY=event.getY();
                break;
            case MotionEvent.ACTION_MOVE:
                endX = event.getX();
                endY = event.getY();
                canvas.drawLine(startX,startY,endX,endY, paint);
                imageView.invalidate();
                startX=endX;
                startY=endY;
                break;
            case MotionEvent.ACTION_UP:
                break;
            case MotionEvent.ACTION_CANCEL:
                break;
            default:
                break;
        }
        return true;
    }

    private Bitmap trim(Bitmap bitmap, int trimColor){
        int minX = Integer.MAX_VALUE;
        int maxX = 0;
        int minY = Integer.MAX_VALUE;
        int maxY = 0;

        for(int x = 0; x < bitmap.getWidth(); x++){
            for(int y = 0; y < bitmap.getHeight(); y++){
                if(bitmap.getPixel(x, y) != trimColor){
                    if(x < minX){
                        minX = x;
                    }
                    if(x > maxX){
                        maxX = x;
                    }
                    if(y < minY){
                        minY = y;
                    }
                    if(y > maxY){
                        maxY = y;
                    }
                }
            }
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        int size = width > height ? width : height;

        if (size == width) {
            size += 1;
            minY = minY - ((size - height) / 2);
            minY = minY < 0 ? 0 : minY;
        } else {
            size += 1;
            minX = minX - ((size - width) / 2);
            minX = minX < 0 ? 0 : minX;
        }

        return Bitmap.createBitmap(bitmap, minX, minY, size, size);
    }

    public static File fileFromAsset(Context context, String assetName) throws IOException {
        File outFile = new File(context.getCacheDir(), assetName );
        copy(context.getAssets().open(assetName), outFile);

        return outFile;
    }

    public static void copy(InputStream inputStream, File output) throws IOException {
        FileOutputStream outputStream = null;

        try {
            outputStream = new FileOutputStream(output);
            boolean read = false;
            byte[] bytes = new byte[1024];

            int read1;
            while((read1 = inputStream.read(bytes)) != -1) {
                outputStream.write(bytes, 0, read1);
            }
        } finally {
            try {
                if(inputStream != null) {
                    inputStream.close();
                }
            } finally {
                if(outputStream != null) {
                    outputStream.close();
                }

            }

        }

    }

    String getAssetFilePath(String name) {
        File tmpFile = null;
        try {
            tmpFile = fileFromAsset(this, name);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        return tmpFile.getAbsolutePath();
    }

    short ARGB8_to_RGB565(int argb)
    {
        if (argb == -1) return 0;

        int a = (argb & 0xFF000000) >> 24;
        int r = (argb & 0x00FF0000) >> 16;
        int g = (argb & 0x0000FF00) >> 8;
        int b = (argb & 0x000000FF);

        r  = r >> 3;
        g  = g >> 2;
        b  = b >> 3;

//        return (short) (b | (g << 5) | (r << (5 + 6)));
        return (short)b;
    }

    int RGB_to_ARGB(int rgb)
    {
        if (rgb == 0) return -1;

        return 0xFF000000 | rgb;
    }

}

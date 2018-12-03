package com.example.leek.my_usb;

import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.hardware.usb.UsbDevice;
import android.os.Build;
import android.os.Environment;
import android.speech.tts.TextToSpeech;
import android.support.annotation.Nullable;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;
import com.jiangdg.usbcamera.UVCCameraHelper;
import com.jiangdg.usbcamera.utils.FileUtils;
import com.serenegiant.usb.common.AbstractUVCCameraHandler;
import com.serenegiant.usb.widget.CameraViewInterface;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {
    private static final String TAG = "Debug";

    /*
    private Thread tts_thread;
    private boolean tts_thread_started = false;
    private boolean feed_bbox_thread_statred = false;
    private boolean program_running = true;

    final int READY_TO_FEED = 0;
    final int SERVICE_CALL_DONE = 1;
    final int TTS_DONE = 2;
    final int WEAK_STATE = 0;
    final int STRONG_STATE = 1;
    final int NORMAL_STATE = 2;
    final int READY_STATE = 3;

    private TextToSpeech tts_object;
    public static String weak_sentence  = "계단이 앞쪽에 있습니다";
    public static String strong_sentence = "계단이 진행방향 바로 앞쪽에 있습니다";
    */

    final int cam_width = 1920;
    final int cam_height = 1080;

    public View mTextureView;
    private boolean isRequest;
    private boolean isPreview;
    private static final int PERMISSION_REQUEST_CODE = 1;

    private ImageView mImageView;
    private static Bitmap bitmap;
    private static Paint  paint;
    private static Canvas canvas;
    private int p_width = 2076;
    private int p_height = 1080;


    String model_name = "mssd_300";
    String model_path ;
    String proto_path = model_path = "/sdcard/saved_images/";
    String device_type = "acl_opencl";

    AlertThread alertThread;

    //temp
    int i = 0;
    int j = 0;

    static long start, end;
    static long e2e_start, e2e_end;
    static long timer[] = new long[10];


    static {
        System.loadLibrary("detect-lib");
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i("onCreate", "1");


        setContentView(R.layout.activity_main);
//        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);


        if (Build.VERSION.SDK_INT >= 23) {
            if (!checkPermission(MainActivity.this)) {
                requestPermission(MainActivity.this); // Code for permission
            }
        }


        boolean create_result = DetectManager.get_graph_space(model_name, model_path + "MobileNetSSD_deploy.caffemodel", proto_path + "MobileNetSSD_deploy.prototxt", device_type);

        // To draw and show BBox
        mImageView = findViewById(R.id.image_view);
        bitmap = Bitmap.createBitmap(p_width, p_height, Bitmap.Config.ARGB_8888);
        mImageView.setImageBitmap(bitmap);


        while (true) {
            byte[] nv21Yuv = new byte[1920 * 1080];

            start = System.currentTimeMillis();
            // Detect BBox
            boolean result = DetectManager.detect(nv21Yuv, 1920, 1080);

            if (result == false)
                Log.i("error", " in obstacle");
            float[] dum = new float[1000];
            DetectManager.get_out_data(dum);

            end = System.currentTimeMillis();
            timer[1] = end - start;  // Detect
            Log.i("night > Detect", "" + timer[1]);

            float x1, y1, x2, y2;
            int n = (int) dum[0];

            start = System.currentTimeMillis();
            // Draw BBox

//            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
            for (int i = 0; i < n; i++) {
                if (dum[1 + i * 6] == 0) {
                    n--;
                    continue;
                }
                // class, state, x1, y1, x2, y2
                x1 = dum[1 + i * 6 + 2] * p_width;
                y1 = dum[1 + i * 6 + 3] * p_height;
                x2 = dum[1 + i * 6 + 4] * p_width;
                y2 = dum[1 + i * 6 + 5] * p_height;
//					canvas.drawRect(x1, y1, x2, y2, paint);
//                canvas.drawRoundRect(x1, y1, x2, y2, 15, 15, paint);

//					if (dum[1 + i*6 + 0] == 20) { //15:person, 20:tvmonitor
//                    Log.i(""+dum[1 + i*6], "x1:"+x1+" y1:"+y1+" x2:"+x2+" y2:"+y2);
//                    Log.i(""+dum[1 + i*6], "x1:"+dum[1 + i*6 + 2]+" y1:"+dum[1 + i*6 + 3]+" x2:"+dum[1 + i*6 + 4]+" y2:"+dum[1 + i*6 + 5]);
//                  }
            }
//
//            if (n == 0) {
//                alertThread.setState(AlertThread.State.NORMAL);
//            } else if (n < 3) {
//                alertThread.setState(AlertThread.State.WARNING);
//            } else {
//                alertThread.setState(AlertThread.State.DANGEROUS);
//            }

            end = System.currentTimeMillis();
            timer[2] = end - start;  // Draw
            Log.i("night > Draw", "" + timer[2]);

            start = System.currentTimeMillis();
            // Release
            DetectManager.delete_out_data();
            end = System.currentTimeMillis();
            timer[3] = end - start;  // Release
            Log.i("night > Release", "" + timer[3]);

            e2e_end = System.currentTimeMillis();
            timer[0] = e2e_end - e2e_start;  // End-to-End

            Log.i("night End-to-End(" + n + ")", "" + timer[0]);

            e2e_start = System.currentTimeMillis();

                /*
                2018-12-03 01:59:55.715 13160-13652/com.example.leek.my_usb I/ >> convert,resize: 12.087
                2018-12-03 01:59:55.715 13160-13652/com.example.leek.my_usb I/ >> normalize: 6.456
                2018-12-03 01:59:55.715 13160-13652/com.example.leek.my_usb I/ >> inference: 81.094
                2018-12-03 01:59:55.715 13160-13652/com.example.leek.my_usb I/ > Detect: 100
                2018-12-03 01:59:55.717 13160-13652/com.example.leek.my_usb I/ > Draw: 2
                2018-12-03 01:59:55.718 13160-13652/com.example.leek.my_usb I/ > Release: 0
                2018-12-03 01:59:55.718 13160-13652/com.example.leek.my_usb I/End-to-End: 115
                 */
        }
    }


    @Override
    protected void onStart() {
        super.onStart();
        Log.i("onStart","3");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.i("onStop","4");
        DetectManager.delete_out_data();

    }



    @Override
    protected void onDestroy() {
        super.onDestroy();
        DetectManager.delete_out_data();
        DetectManager.delete_graph_space();
    }

    private void showShortMsg(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }


    private boolean checkPermission(Context context) {//
        int result = ContextCompat.checkSelfPermission(context, android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (result == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            return false;
        }
    }
    private void requestPermission(Activity activity) {

        if (ActivityCompat.shouldShowRequestPermissionRationale(activity, android.Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            Toast.makeText(activity, "Write External Storage permission allows us to do store images. Please allow this permission in App Settings.", Toast.LENGTH_LONG).show();
        } else {
            ActivityCompat.requestPermissions(activity, new String[]{android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        }
    }
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.e("value", "Permission Granted, Now you can use local drive .");
                } else {
                    Log.e("value", "Permission Denied, You cannot use local drive .");
                }
                break;
        }
    }

}

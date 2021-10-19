package jp.co.people.camerax_sample

import android.content.res.AssetFileDescriptor
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Executors.newSingleThreadExecutor
import java.util.jar.Manifest

class MainActivity : AppCompatActivity() {

    private lateinit var cameraExecutor: ExecutorService
    private lateinit var overlaySurfaceView: OverlaySurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        overlaySurfaceView = OverlaySurfaceView(resultView)

        cameraExecutor = Executors.newSingleThreadExecutor()
        setupCamera()

    }


    fun setupCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(cameraView.surfaceProvider) }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // TODO 物体検知
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetRotation(cameraView.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(
                        cameraExecutor,
                        ObjectDetector(
                            yuvToRgbConverter,
                            interpreter,
                            labels,
                            Size(resultView.width, resultView.height)
                        ) {
                            detectedObjectList ->
                            // TODO 検出結果の表示
                            overlaySurfaceView.draw(detectedObjectList)
                        }
                    )
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: java.lang.Exception) {
                Log.e("ERROR: Camera", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val MODEL_FILE_NAME = "detect.tflite"
        private const val LABEL_FILE_NAME = "labelmap.txt"
    }

    private val yuvToRgbConverter: YuvToRgbConverter by lazy {
        YuvToRgbConverter(this)
    }

    private val interpreter: Interpreter by lazy {
        Interpreter(loadModel())
    }

    private val labels: List<String> by lazy {
        loadLabels()
    }

    private fun loadModel(fileName: String = MainActivity.MODEL_FILE_NAME): ByteBuffer {
        lateinit var modelBuffer: ByteBuffer
        var file: AssetFileDescriptor? = null
        try {
            file = assets.openFd(fileName)
            val inputStream = FileInputStream(file.fileDescriptor)
            val fileChannel = inputStream.channel
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, file.startOffset, file.declaredLength)
        } catch (e: Exception) {
            Toast.makeText(this, "model load error", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            file?.close()
        }
        return modelBuffer
    }

    private fun loadLabels(fileName: String = MainActivity.LABEL_FILE_NAME): List<String> {
        var labels = listOf<String>()
        var inputStream: InputStream? = null
        try {
            inputStream = assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            labels = reader.readLines()
        } catch (e: Exception) {
            Toast.makeText(this, "label load error", Toast.LENGTH_SHORT).show()
            finish()
        } finally {
            inputStream?.close()
        }
        return labels
    }

}

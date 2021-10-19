package jp.co.people.camerax_sample

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.util.Size
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.internal.YuvToJpegProcessor
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.text.Normalizer

typealias ObjectDetectorCallback = (image: List<DetectionObject>) -> Unit

class ObjectDetector (
    private val yuvToRgbConverter: YuvToRgbConverter,
    private val interpreter: Interpreter,
    private val labels: List<String>,
    private val resultViewSize: Size,
    private val listener: ObjectDetectorCallback
) : ImageAnalysis.Analyzer {

    companion object {
        // モデル のinputとoutputサイズ
        private const val IMG_SIZE_X = 300
        private const val IMG_SIZE_Y = 300
        private const val MAX_DETECTION_NUM = 10

        // 今回使うtfliteモデルは量子化済みなので、normalize関連は 127.5f ではなく以下の通り
        private const val NORMALIZE_MEAN = 0f
        private const val NORMALIZE_STD = 1f

        // 検出結果のスコアしきい値
        private const val SCORE_THRESHOLD = 0.5f
    }

    private var imageRotationDegrees: Int = 0
    private val tfImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(IMG_SIZE_X, IMG_SIZE_Y, ResizeOp.ResizeMethod.BILINEAR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD))
            .build()
    }
    private val tfImageBuffer = TensorImage(DataType.UINT8)

    private val outputBoundingBoxes: Array<Array<FloatArray>> = arrayOf(
        Array(MAX_DETECTION_NUM) {
            FloatArray(4)
        }
    )

    private val outputLabels: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )

    private val outputScores: Array<FloatArray> = arrayOf(
        FloatArray(MAX_DETECTION_NUM)
    )
    private val outputDetectionNum: FloatArray = FloatArray(1)

    private val outputMap = mapOf(
        0 to outputBoundingBoxes,
        1 to outputLabels,
        2 to outputScores,
        3 to outputDetectionNum
    )

    private fun detect(targetImage: Image): List<DetectionObject> {
        val targetBitmap = Bitmap.createBitmap(targetImage.width, targetImage.height, Bitmap.Config.ARGB_8888)
        yuvToRgbConverter.yuvToRgb(targetImage, targetBitmap)
        tfImageBuffer.load(targetBitmap)
        val tensorImage = tfImageProcessor.process(tfImageBuffer)

        interpreter.runForMultipleInputsOutputs(arrayOf(tensorImage.buffer), outputMap)

        val detectedObjectList = arrayListOf<DetectionObject>()
        loop@ for (i in 0 until outputDetectionNum[0].toInt()) {
            val score = outputScores[0][i]
            val label = labels[outputLabels[0][i].toInt()]
            val boundingBox = RectF(
                outputBoundingBoxes[0][i][1] * resultViewSize.width,
                outputBoundingBoxes[0][i][0] * resultViewSize.height,
                outputBoundingBoxes[0][i][3] * resultViewSize.width,
                outputBoundingBoxes[0][i][2] * resultViewSize.height
            )
            if (score >= ObjectDetector.SCORE_THRESHOLD) {
                detectedObjectList.add(
                    DetectionObject(
                        score = score,
                        label = label,
                        boundingBox = boundingBox
                    )
                )
            } else {
                // 検出結果はスコアの高い順にソートされているので、下回ったら終了
                break@loop
            }
        }
        return detectedObjectList.take(4)
    }
    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(image: ImageProxy) {
        if (image.image == null) return
        imageRotationDegrees = image.imageInfo.rotationDegrees
        val detectedObjectList = detect(image.image!!)
        listener(detectedObjectList)
        image.close()
    }
}

/**
 * 検出結果を入れるクラス
 */
data class DetectionObject(
    val score: Float,
    val label: String,
    val boundingBox: RectF
)



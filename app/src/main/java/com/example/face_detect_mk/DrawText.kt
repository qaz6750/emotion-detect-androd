package com.example.face_detect_mk

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Canvas
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Paint
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.unit.sp

/**
 * DrawScope的扩展函数，用于在Canvas上绘制文本
 */
fun DrawScope.drawText(
    text: String,
    x: Float,
    y: Float,
    color: Color,
    textSize: Float = 36f // 默认文本大小
) {
    val paint = Paint().asFrameworkPaint().apply {
        this.color = color.toArgb()
        this.textSize = textSize
        this.isAntiAlias = true
    }
    
    drawContext.canvas.nativeCanvas.drawText(text, x, y, paint)
}

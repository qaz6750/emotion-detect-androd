package com.example.face_detect_mk

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.face_detect_mk.ui.theme.Face_detect_MKTheme

/**
 * 主活动类
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Face_detect_MKTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    FaceDetectionApp(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

/**
 * 应用导航状态
 */
enum class AppScreen {
    MAIN,
    REAL_TIME_BACK
}

/**
 * 主应用界面，包含标签页和全屏后置摄像头模式
 */
@Composable
fun FaceDetectionApp(modifier: Modifier = Modifier) {
    var currentScreen by remember { mutableStateOf(AppScreen.MAIN) }
    
    when (currentScreen) {
        AppScreen.MAIN -> {
            MainScreen(
                modifier = modifier,
                onBackCameraClick = { currentScreen = AppScreen.REAL_TIME_BACK }
            )
        }
        AppScreen.REAL_TIME_BACK -> {
            RealTimeBackFaceDetection(
                onBack = { currentScreen = AppScreen.MAIN }
            )
        }
    }
}

/**
 * 主屏幕界面
 */
@Composable
fun MainScreen(
    modifier: Modifier = Modifier,
    onBackCameraClick: () -> Unit
) {
    var selectedTabIndex by remember { mutableIntStateOf(0) }
    
    Column(
        modifier = modifier.fillMaxSize()
    ) {
        Text(
            text = "Emotion Detection", 
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.padding(16.dp)
        )
        
        // 摄像头模式选择按钮
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {            Button(
                onClick = onBackCameraClick,
                modifier = Modifier.weight(1f)
            ) {
                Text("后置摄像头全屏检测")
            }
        }
        
        Spacer(modifier = Modifier.height(16.dp))
        
        TabRow(selectedTabIndex = selectedTabIndex) {            Tab(
                selected = selectedTabIndex == 0,
                onClick = { selectedTabIndex = 0 },
                text = { 
                    Text("图片检测")
                }
            )
            Tab(
                selected = selectedTabIndex == 1,
                onClick = { selectedTabIndex = 1 },
                text = { 
                    Text("前置实时检测")
                }
            )
        }
        
        when (selectedTabIndex) {
            0 -> ImageFaceDetection()
            1 -> RealTimeFaceDetection()
        }
    }
}
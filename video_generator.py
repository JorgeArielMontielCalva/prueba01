"""
Módulo para generar videos con movimiento a partir de imágenes
Soporta dos tipos: Panorámico (pan & zoom) y Acción (movimiento dinámico)
"""

import cv2
import numpy as np
from pathlib import Path
from config import VIDEO_FORMATS, MOTION_CONFIG, VIDEO_QUALITY
import subprocess
import os
import shutil


def _read_image_unicode_safe(image_path):
    """Lee imágenes con soporte para rutas Unicode en Windows."""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class VideoGenerator:
    def __init__(self):
        """Inicializa el generador de videos"""
        self.video_formats = VIDEO_FORMATS
        self.motion_config = MOTION_CONFIG

    def _sanitize_name(self, name):
        """Limpia un nombre para usarlo en archivos de salida."""
        invalid_chars = '<>:"/\\|?*'
        cleaned = ''.join('_' if char in invalid_chars else char for char in name.strip())
        return cleaned.rstrip(' .') or 'video'
    
    def load_and_resize_image(self, image_path, target_size):
        """Carga y redimensiona imagen para video"""
        image = _read_image_unicode_safe(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Obtener dimensiones originales
        height, width = image.shape[:2]
        
        # Calcular escala para "zoomed in" (versión ampliada)
        zoom_scale = target_size['zoom_factor']
        new_width = int(width * zoom_scale)
        new_height = int(height * zoom_scale)
        
        # Redimensionar imagen ampliada
        image_zoomed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return image, image_zoomed, (width, height)
    
    def create_panoramic_motion(self, image_full, image_zoomed, format_config, duration, motion_settings):
        """Crea movimiento cinematográfico A-B (panorámico)"""
        width = format_config['width']
        height = format_config['height']
        fps = format_config['fps']
        total_frames = int(fps * duration)
        
        frames = []
        
        # Movimiento suave de pan y zoom - estilo cinematográfico
        for frame_idx in range(total_frames):
            progress = frame_idx / total_frames
            
            # Crear interpolación suave (ease-in-out)
            t = progress
            smooth_t = t * t * (3 - 2 * t)  # smoothstep
            
            # Crear marco interpolado
            frame = self._interpolate_frame_panoramic(
                image_full, image_zoomed, 
                width, height, 
                smooth_t, motion_settings
            )
            
            frames.append(frame)
        
        return frames
    
    def create_action_motion(self, image_full, image_zoomed, format_config, duration, motion_settings):
        """Crea movimiento dinámico simulando actividad (acción)"""
        width = format_config['width']
        height = format_config['height']
        fps = format_config['fps']
        total_frames = int(fps * duration)
        
        frames = []
        
        # Movimiento más dinámico con múltiples puntos de interés
        for frame_idx in range(total_frames):
            progress = frame_idx / total_frames
            
            # Crear movimientos variados simulando energía
            frame = self._interpolate_frame_action(
                image_full, image_zoomed,
                width, height,
                progress, frame_idx, total_frames, motion_settings
            )
            
            frames.append(frame)
        
        return frames
    
    def _interpolate_frame_panoramic(self, image_full, image_zoomed, width, height, t, motion_settings):
        """Interpola frame para movimiento panorámico suave"""
        img_h, img_w = image_full.shape[:2]
        
        # Calcular zoom interpolado
        zoom_start = 1.0
        zoom_end = motion_settings['zoom_factor']
        zoom = zoom_start + (zoom_end - zoom_start) * t
        
        # Redimensionar imagen actual
        resized_width = int(img_w * zoom)
        resized_height = int(img_h * zoom)
        current_image = cv2.resize(image_full, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Movimiento de pan
        pan_intensity = motion_settings['pan_intensity']
        pan_x = int((resized_width - width) * t * pan_intensity)
        pan_y = int((resized_height - height) * (0.5 + 0.5 * np.sin(t * np.pi)))
        
        # Asegurar que pan no sale de límites
        pan_x = max(0, min(pan_x, resized_width - width))
        pan_y = max(0, min(pan_y, resized_height - height))
        
        # Extraer región
        frame = current_image[pan_y:pan_y+height, pan_x:pan_x+width]
        
        # Rellenar si es necesario
        if frame.shape[:2] != (height, width):
            frame = self._pad_frame(frame, width, height)
        
        return frame
    
    def _interpolate_frame_action(self, image_full, image_zoomed, width, height, progress, frame_idx, total_frames, motion_settings):
        """Interpola frame para movimiento de acción"""
        # Movimiento más dinámico con múltiples puntos
        
        # Usar oscilación suave para crear sensación de movimiento
        oscillation = np.sin(frame_idx / 12.0) * 0.15
        zoom = 1.0 + (motion_settings['zoom_factor'] - 1.0) * (0.35 + 0.25 * np.sin(progress * np.pi * 2))
        
        img_h, img_w = image_full.shape[:2]
        resized_width = int(img_w * zoom)
        resized_height = int(img_h * zoom)
        
        current_image = cv2.resize(image_full, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Pan más dinámico
        pan_x = int((resized_width - width) * (0.5 + 0.15 * np.sin(progress * np.pi * 4 + oscillation)))
        pan_y = int((resized_height - height) * (0.5 + 0.12 * np.cos(progress * np.pi * 3 + oscillation)))
        
        pan_x = max(0, min(pan_x, resized_width - width))
        pan_y = max(0, min(pan_y, resized_height - height))
        
        frame = current_image[pan_y:pan_y+height, pan_x:pan_x+width]
        
        if frame.shape[:2] != (height, width):
            frame = self._pad_frame(frame, width, height)
        
        return frame
    
    def _pad_frame(self, frame, target_width, target_height):
        """Rellena un frame con diferencias de tamaño"""
        h, w = frame.shape[:2]
        
        if h == target_height and w == target_width:
            return frame
        
        # Crear nuevo frame del tamaño correcto
        new_frame = np.zeros((target_height, target_width, frame.shape[2] if len(frame.shape) > 2 else 1), dtype=frame.dtype)
        
        # Calcular posición central
        y_offset = (target_height - h) // 2
        x_offset = (target_width - w) // 2
        
        # Colocar imagen en el centro
        new_frame[y_offset:y_offset+h, x_offset:x_offset+w] = frame
        
        return new_frame
    
    def generate_video(self, image_path, output_path, video_type, format_type='vertical', duration=5):
        """
        Genera un video a partir de una imagen
        
        Args:
            image_path: Ruta a la imagen
            output_path: Ruta de salida del video
            video_type: 'panoramic' o 'action'
            format_type: 'vertical' o 'square'
            duration: Duración en segundos
        """
        try:
            # Obtener configuración
            format_config = self.video_formats[format_type]
            motion_settings = self.motion_config[video_type]
            
            print(f"Generando video {video_type} ({format_type}): {image_path}")
            
            # Cargar imagen
            image_full, image_zoomed, original_size = self.load_and_resize_image(
                image_path, motion_settings
            )
            
            # Generar frames según tipo de video
            if video_type == 'panoramic':
                frames = self.create_panoramic_motion(
                    image_full, image_zoomed, format_config, duration, motion_settings
                )
            else:  # action
                frames = self.create_action_motion(
                    image_full, image_zoomed, format_config, duration, motion_settings
                )
            
            # Escribir video con ffmpeg para máxima calidad
            self._write_video_ffmpeg(frames, output_path, format_config)
            
            print(f"Video generado: {output_path}")
            return True
        
        except Exception as e:
            print(f"Error generando video: {str(e)}")
            return False
    
    def _write_video_ffmpeg(self, frames, output_path, format_config):
        """Escribe video con FFmpeg empaquetado y usa OpenCV solo como último recurso."""
        fps = format_config['fps']
        width = format_config['width']
        height = format_config['height']

        ffmpeg_path = None
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = shutil.which('ffmpeg')

        if ffmpeg_path:
            command = [
                ffmpeg_path,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',
                '-an',
                '-c:v', VIDEO_QUALITY['codec'],
                '-preset', VIDEO_QUALITY['preset'],
                '-crf', str(VIDEO_QUALITY['crf']),
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                output_path,
            ]

            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stderr_data = b''
            try:
                for frame in frames:
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    process.stdin.write(frame.tobytes())
                process.stdin.close()
                _, stderr_data = process.communicate()
                if process.returncode != 0:
                    raise ValueError(stderr_data.decode('utf-8', errors='ignore'))
                return
            except Exception as error:
                if process.stdin and not process.stdin.closed:
                    process.stdin.close()
                process.kill()
                raise ValueError(f"FFmpeg falló: {error}")

        raise ValueError("No se encontró FFmpeg funcional para exportar el video con calidad alta")
    
    def generate_all_formats(self, image_path, video_type, output_base_path, base_name=None):
        """Genera video en todos los formatos"""
        output_name = self._sanitize_name(base_name or Path(image_path).stem)
        success = True
        
        for index, format_type in enumerate(['vertical', 'square'], 1):
            format_config = self.video_formats[format_type]
            dimensions = f"{format_config['width']}x{format_config['height']}"
            output_path = f"{output_base_path}/{output_name}_{index}_{dimensions}.mp4"
            
            if not self.generate_video(image_path, output_path, video_type, format_type):
                success = False
        
        return success

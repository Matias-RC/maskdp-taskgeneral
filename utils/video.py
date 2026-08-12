import cv2
import imageio
import numpy as np
import torch
from typing import Optional, Dict, Any

class VideoRecorder:
    def __init__(self,
                 root_dir,
                 render_size=256,
                 fps=20,
                 camera_id=0):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True, parents=True) 
        else:
            self.save_dir = None
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if self.enabled:
            self.record(env)

    def record(self, env, metadata: Optional[Dict[str, Any]] = None):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=self.camera_id)
            else:
                frame = env.render()
            frame = frame.copy()
            if metadata:
                frame = self.overlay_metadata(frame, metadata)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled and len(self.frames) > 0:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
    
    def render_goal(self, env, goal_physics):
        """Renderiza el estado objetivo y lo añade al final del video como un fotograma estático."""
        if not self.enabled:
            return
            
        if torch.is_tensor(goal_physics):
            goal_state = goal_physics.cpu().numpy()
        else:
            goal_state = np.asarray(goal_physics)
            
        with env.physics.reset_context():
            env.physics.set_state(goal_state)
            
        frame = env.physics.render(height=self.render_size,
                                   width=self.render_size,
                                   camera_id=self.camera_id)
        frame = frame.copy()

        cv2.putText(frame, "GOAL STATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        hold_frames = self.fps * 2
        for _ in range(hold_frames):
            self.frames.append(frame)

    def overlay_metadata(self, frame, metadata):
        cv2.rectangle(frame, (156, 0), (256, 45), (0, 0, 0), -1)
        start_x = 162 
        start_y = 15  
        line_spacing = 15
        for i, (key, value) in enumerate(metadata.items()):
            text = f"{key}: {value}"
            current_y = start_y + (i * line_spacing)
            cv2.putText(frame, text, (start_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return frame
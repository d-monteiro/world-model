"""
Renderer for 2D Robotic Arm Environment

Provides visualization using matplotlib for both human viewing and image generation.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
import io
from PIL import Image


class ArmRenderer:
    """Renderer for the 2D robotic arm environment."""
    
    def __init__(
        self,
        workspace_size: float = 2.0,
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 100
    ):
        """
        Initialize the renderer.
        
        Args:
            workspace_size: Size of the workspace
            figsize: Figure size in inches
            dpi: Dots per inch for rendering
        """
        self.workspace_size = workspace_size
        self.figsize = figsize
        self.dpi = dpi
        
        self.fig = None
        self.ax = None
        self.initialized = False
    
    def initialize(self):
        """Initialize matplotlib figure and axes."""
        if not self.initialized:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            self.fig.canvas.manager.set_window_title('2D Robotic Arm Environment')
            self.ax.set_xlim(-self.workspace_size, self.workspace_size)
            self.ax.set_ylim(-self.workspace_size, self.workspace_size)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title('2D Robotic Arm Environment')
            plt.show(block=False)
            self.initialized = True
    
    def render(
        self,
        joint_angles: np.ndarray,
        link_lengths: np.ndarray,
        object_positions: np.ndarray,
        goal_positions: np.ndarray,
        grasped_object: int = -1,
        show: bool = True
    ) -> Optional[np.ndarray]:
        """
        Render the current state of the environment.
        
        Args:
            joint_angles: Array of joint angles
            link_lengths: Array of link lengths
            object_positions: Array of object positions (N, 2)
            goal_positions: Array of goal positions (N, 2)
            grasped_object: Index of grasped object (-1 if none)
            show: Whether to display the plot
            
        Returns:
            RGB array if show=False, None otherwise
        """
        self.initialize()
        self.ax.clear()
        
        # Set up plot
        self.ax.set_xlim(-self.workspace_size, self.workspace_size)
        self.ax.set_ylim(-self.workspace_size, self.workspace_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('2D Robotic Arm Environment')
        
        # Draw arm
        self._draw_arm(joint_angles, link_lengths)
        
        # Draw objects
        self._draw_objects(object_positions, goal_positions, grasped_object)
        
        # Draw goals
        self._draw_goals(goal_positions)
        
        if show:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.05)  # Small pause for animation (50ms)
            return None
        else:
            # Convert to RGB array
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            rgb_array = np.array(img)
            return rgb_array
    
    def _draw_arm(self, joint_angles: np.ndarray, link_lengths: np.ndarray):
        """Draw the robotic arm."""
        x, y = 0.0, 0.0  # Base position
        angle_sum = 0.0
        
        # Draw base
        self.ax.plot([0], [0], 'ko', markersize=10, label='Base')
        
        # Draw each link
        for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
            angle_sum += angle
            next_x = x + length * np.cos(angle_sum)
            next_y = y + length * np.sin(angle_sum)
            
            # Draw link
            self.ax.plot([x, next_x], [y, next_y], 'b-', linewidth=3, alpha=0.7)
            
            # Draw joint
            self.ax.plot([x], [y], 'ro', markersize=8)
            
            x, y = next_x, next_y
        
        # Draw end-effector
        self.ax.plot([x], [y], 'go', markersize=10, label='End-effector')
    
    def _draw_objects(
        self,
        object_positions: np.ndarray,
        goal_positions: np.ndarray,
        grasped_object: int
    ):
        """Draw movable objects."""
        for i, obj_pos in enumerate(object_positions):
            color = 'orange' if i == grasped_object else 'blue'
            circle = patches.Circle(
                obj_pos,
                radius=0.1,
                color=color,
                alpha=0.7,
                label='Object' if i == 0 else None
            )
            self.ax.add_patch(circle)
            
            # Draw object center
            self.ax.plot([obj_pos[0]], [obj_pos[1]], 'k.', markersize=3)
    
    def _draw_goals(self, goal_positions: np.ndarray):
        """Draw goal positions."""
        for i, goal_pos in enumerate(goal_positions):
            circle = patches.Circle(
                goal_pos,
                radius=0.1,
                color='green',
                alpha=0.3,
                linestyle='--',
                fill=False,
                linewidth=2,
                label='Goal' if i == 0 else None
            )
            self.ax.add_patch(circle)
            
            # Draw goal center
            self.ax.plot([goal_pos[0]], [goal_pos[1]], 'gx', markersize=10, markeredgewidth=2)
    
    def close(self):
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.initialized = False


def render_to_image(
    joint_angles: np.ndarray,
    link_lengths: np.ndarray,
    object_positions: np.ndarray,
    goal_positions: np.ndarray,
    workspace_size: float = 2.0,
    image_size: Tuple[int, int] = (64, 64),
    grasped_object: int = -1
) -> np.ndarray:
    """
    Render environment state to a small image (for image-based VAE).
    
    Args:
        joint_angles: Array of joint angles
        link_lengths: Array of link lengths
        object_positions: Array of object positions
        goal_positions: Array of goal positions
        workspace_size: Size of workspace
        image_size: Output image size (height, width)
        grasped_object: Index of grasped object
        
    Returns:
        RGB image array of shape (H, W, 3)
    """
    renderer = ArmRenderer(workspace_size=workspace_size, figsize=(4, 4), dpi=image_size[0] // 4)
    renderer.initialize()
    
    rgb_array = renderer.render(
        joint_angles=joint_angles,
        link_lengths=link_lengths,
        object_positions=object_positions,
        goal_positions=goal_positions,
        grasped_object=grasped_object,
        show=False
    )
    
    # Resize to desired image size
    img = Image.fromarray(rgb_array)
    img = img.resize(image_size, Image.Resampling.LANCZOS)
    rgb_array = np.array(img)
    
    renderer.close()
    
    return rgb_array

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

class LanguageCommander:
    def __init__(self, workspace_radius=1.5):
        self.radius = workspace_radius
        self.model = None
        self.commands = {}
        self.command_embeddings = None
        
        if HAS_NLP:
            # Load quietly
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._init_commands()

    def _init_commands(self):
        r = self.radius * 0.8
        
        self.commands = {
            "center":       [0.0, 0.0],
            "go home":      [0.0, 0.0],
            "top":          [0.0, r],
            "up":           [0.0, r],
            "north":        [0.0, r],
            "bottom":       [0.0, -r],
            "down":         [0.0, -r],
            "south":        [0.0, -r],
            "right":        [r, 0.0],
            "east":         [r, 0.0],
            "left":         [-r, 0.0],
            "west":         [-r, 0.0],
            "top right":    [r/1.4, r/1.4],
            "top left":     [-r/1.4, r/1.4],
            "bottom right": [r/1.4, -r/1.4],
            "bottom left":  [-r/1.4, -r/1.4],
        }
        
        self.command_texts = list(self.commands.keys())
        self.command_embeddings = self.model.encode(self.command_texts, convert_to_tensor=True)

    def get_goal(self, text):
        """
        Returns (x, y) target coordinate for the given text command.
        """
        if not HAS_NLP or self.model is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        user_emb = self.model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(user_emb, self.command_embeddings)[0]
        best_idx = torch.argmax(scores).item()
        
        match_text = self.command_texts[best_idx]
        return np.array(self.commands[match_text], dtype=np.float32)

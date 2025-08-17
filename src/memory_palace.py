import random
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch
import numpy as np

try:
    from skrl.memories.torch import Memory
except ImportError:
    # Fallback if SKRL is not available
    class Memory:
        def __init__(self, memory_size: int, num_envs: int = 1, device: Optional[Union[str, torch.device]] = None,
                     export: bool = False, export_format: str = "pt", export_directory: str = ""):
            self.memory_size = memory_size
            self.num_envs = num_envs
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.export = export
            self.export_format = export_format
            self.export_directory = export_directory
            self.tensors = {}
            self.filled = False
            self.sampling_indexes = None

        def add_tensors(self, tensors_names: List[str]) -> None:
            for name in tensors_names:
                self.tensors[name] = torch.empty((self.memory_size, self.num_envs), device=self.device)

        def __len__(self) -> int:
            return self.memory_size if self.filled else 0

        def sample_by_index(self, names: Tuple[str], indexes: torch.Tensor, mini_batches: int = 1) -> List[List[torch.Tensor]]:
            return [[self.tensors[name][indexes] for name in names]]


class MemoryPalace(Memory):
    """
    Memory Palace for phase-action balanced experience replay.
    
    Each unique action has its own memory buffer for balanced sampling.
    Compatible with SKRL Memory interface.
    """

    def __init__(
        self,
        memory_size: int,
        num_envs: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        export: bool = False,
        export_format: str = "pt",
        export_directory: str = "",
        max_size_per_palace: Optional[int] = None,
        min_size_to_sample: int = 10,
        replacement: bool = True,
    ) -> None:
        """
        Args:
            memory_size (int): Maximum total number of elements across all palaces
            num_envs (int): Number of parallel environments (default: 1)
            device: Device on which tensors are allocated
            export (bool): Export the memory to a file (default: False)
            export_format (str): Export format (default: "pt")
            export_directory (str): Directory where the memory will be exported (default: "")
            max_size_per_palace (int): Max size for each individual memory buffer (default: memory_size // 4)
            min_size_to_sample (int): Minimum size needed to sample from a palace (default: 10)
            replacement (bool): Whether to sample with replacement (default: True)
        """
        super().__init__(memory_size, num_envs, device, export, export_format, export_directory)

        self.max_size_per_palace = max_size_per_palace or max(memory_size // 4, 100)
        self.min_size_to_sample = min_size_to_sample
        self.replacement = replacement
        
        # Each palace stores transitions for a specific action
        self.palaces = defaultdict(list)  # {action: [transitions]}
        self.palace_sizes = defaultdict(int)  # {action: current_size}
        
        # Track total size across all palaces
        self.total_size = 0
        self.position = 0
        
        # Track tensor names for SKRL compatibility
        self.tensor_names = []

    def create_tensor(self, name: str, size: Union[int, Tuple[int]], dtype: torch.dtype) -> None:
        """
        Create a tensor in memory (SKRL compatibility method).
        
        Args:
            name: Name of the tensor
            size: Size of the tensor 
            dtype: Data type of the tensor
        """
        # For MemoryPalace, we don't need to pre-allocate tensors like SKRL's RandomMemory
        # We just track the tensor names and their expected dtypes
        if name not in self.tensor_names:
            self.tensor_names.append(name)
        
        # Store tensor metadata for validation
        if not hasattr(self, '_tensor_metadata'):
            self._tensor_metadata = {}
        
        self._tensor_metadata[name] = {
            'size': size,
            'dtype': dtype
        }

    def get_tensor_names(self) -> List[str]:
        """Get list of tensor names for SKRL compatibility"""
        return self.tensor_names.copy()

    def add_samples(self, **kwargs) -> None:
        """
        Add samples to memory (SKRL compatibility method).
        
        This method converts SKRL's add_samples interface to our add method.
        """
        # Convert SKRL format to our internal format
        transitions = {}
        
        for name, tensor in kwargs.items():
            if isinstance(tensor, torch.Tensor):
                # Remove batch dimension if present (SKRL adds batch dim)
                if tensor.dim() > 1 and tensor.shape[0] == 1:
                    transitions[name] = tensor.squeeze(0)
                else:
                    transitions[name] = tensor
            else:
                transitions[name] = tensor
        
        # Use our add method
        self.add(transitions)

    def _get_palace_key(self, action: Union[int, torch.Tensor]) -> int:
        """Define how to key the palace - here we use the action value"""
        if isinstance(action, torch.Tensor):
            return action.item()
        return int(action)

    def add(self, tensors: dict, **kwargs) -> None:
        """
        Add a transition to the appropriate memory palace.
        
        Args:
            tensors (dict): Dictionary containing transition data
                          Expected keys: 'states', 'actions', 'rewards', 'next_states', 'terminated', 'truncated'
        """
        # Extract action to determine which palace to use
        action = tensors.get('actions')
        if action is None:
            raise ValueError("Action must be provided in tensors dict")
        
        palace_key = self._get_palace_key(action)
        palace = self.palaces[palace_key]
        
        # Create transition tuple compatible with typical RL format
        transition = {
            'states': tensors.get('states'),
            'actions': tensors.get('actions'), 
            'rewards': tensors.get('rewards'),
            'next_states': tensors.get('next_states'),
            'terminated': tensors.get('terminated', False),
            'truncated': tensors.get('truncated', False)
        }
        
        # Add to palace
        palace.append(transition)
        self.palace_sizes[palace_key] += 1
        self.total_size += 1
        
        # Remove oldest if palace exceeds max size
        if len(palace) > self.max_size_per_palace:
            palace.pop(0)
            self.palace_sizes[palace_key] -= 1
            self.total_size -= 1

    def sample(
        self, 
        names: Tuple[str], 
        batch_size: int, 
        mini_batches: int = 1, 
        sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """
        Sample a batch from memory palaces with balanced sampling.
        
        Args:
            names: Tensor names from which to obtain samples
            batch_size: Number of elements to sample
            mini_batches: Number of mini-batches to sample (default: 1)
            sequence_length: Length of each sequence (default: 1)
            
        Returns:
            Sampled data from tensors sorted according to their position in the list of names
        """
        if sequence_length > 1:
            raise NotImplementedError("Sequence sampling not implemented for MemoryPalace")
        
        # Get valid palaces (those with enough samples)
        valid_palaces = {k: v for k, v in self.palaces.items() 
                        if len(v) >= self.min_size_to_sample}
        
        if not valid_palaces:
            # If no valid palaces, sample from all available data
            all_transitions = []
            for palace in self.palaces.values():
                all_transitions.extend(palace)
            
            if not all_transitions:
                # Return empty tensors if no data
                return [[torch.empty((0,) + self.tensors.get(name, torch.empty(0)).shape[1:], 
                                   device=self.device) for name in names] for _ in range(mini_batches)]
            
            # Sample from all transitions
            if self.replacement:
                sampled_transitions = random.choices(all_transitions, k=batch_size)
            else:
                sampled_transitions = random.sample(all_transitions, 
                                                  min(batch_size, len(all_transitions)))
        else:
            # Balanced sampling from valid palaces
            samples_per_palace = max(1, batch_size // len(valid_palaces))
            remaining_samples = batch_size - (samples_per_palace * len(valid_palaces))
            
            sampled_transitions = []
            palace_keys = list(valid_palaces.keys())
            
            for i, (palace_key, palace) in enumerate(valid_palaces.items()):
                # Calculate samples for this palace
                current_samples = samples_per_palace
                if i < remaining_samples:
                    current_samples += 1
                
                # Sample from this palace
                if self.replacement:
                    palace_samples = random.choices(palace, k=min(current_samples, len(palace)))
                else:
                    palace_samples = random.sample(palace, min(current_samples, len(palace)))
                
                sampled_transitions.extend(palace_samples)
        
        # Convert transitions to tensor format expected by SKRL
        batch_data = {name: [] for name in names}
        
        for transition in sampled_transitions:
            for name in names:
                if name in transition and transition[name] is not None:
                    batch_data[name].append(transition[name])
        
        # Convert to tensors and create mini-batches
        result = []
        for _ in range(mini_batches):
            mini_batch = []
            for name in names:
                if batch_data[name]:
                    # Stack tensors for this name
                    tensor_data = torch.stack([torch.as_tensor(item, device=self.device) 
                                             for item in batch_data[name]])
                    mini_batch.append(tensor_data)
                else:
                    # Empty tensor if no data for this name
                    mini_batch.append(torch.empty((len(sampled_transitions), 0), device=self.device))
            result.append(mini_batch)
        
        return result

    def __len__(self) -> int:
        """Return total number of transitions across all palaces"""
        return self.total_size

    def get_palace_stats(self) -> dict:
        """Get statistics about palace usage"""
        return {
            'total_palaces': len(self.palaces),
            'palace_sizes': dict(self.palace_sizes),
            'total_size': self.total_size,
            'valid_palaces': len([p for p in self.palaces.values() 
                                if len(p) >= self.min_size_to_sample])
        }

    def reset(self) -> None:
        """Clear all palaces"""
        self.palaces = defaultdict(list)
        self.palace_sizes = defaultdict(int)
        self.total_size = 0
        self.position = 0

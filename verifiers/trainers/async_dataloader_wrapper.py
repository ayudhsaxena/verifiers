from typing import Any, List
from torch.utils.data import DataLoader #type: ignore
from collections import deque
import threading


class AsyncDataLoaderWrapper:
    """
    Wraps a DataLoader to provide batch prefetching capabilities for async generation.
    
    This wrapper maintains a buffer of upcoming batches that can be accessed
    without advancing the main iterator, allowing async generation to work
    ahead while training continues on current batches.
    """

    def __init__(self, dataloader: DataLoader, buffer_size: int = 5):
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._current_iterator = None  # Iterator for current epoch
        self._next_iterator = None     # Iterator for next epoch (created when needed)
        self._lock = threading.Lock()
        self._exhausted = False
        self._current_batch = None  # Track the current batch
        
    def __iter__(self):
        """Reset and return iterator"""
        with self._lock:
            # If we pre-created an iterator for the next epoch, use it
            if self._next_iterator is not None:
                self._current_iterator = self._next_iterator
                self._next_iterator = None
            else:
                self._current_iterator = iter(self.dataloader)
            
            self._buffer.clear()
            self._exhausted = False
            self._current_batch = None
        return self
        
    def __next__(self):
        """Get next batch, refilling buffer as needed"""
        with self._lock:
            # If buffer is empty, try to fill it
            if not self._buffer and not self._exhausted:
                self._fill_buffer()
                
            if not self._buffer:
                raise StopIteration
                
            # Store current batch before returning
            self._current_batch = self._buffer.popleft()
            return self._current_batch

    def peek_ahead(self, n: int = 1) -> List[Any]:
        """
        Peek at the next *n* batches without consuming them.
        Special-case *n == 0*: this is interpreted as *"return the current batch"*.
        If no batch has been consumed yet, we transparently fetch the first batch so
        the caller never receives an empty list.
        """
        with self._lock:
            if n == 0:
                # If no current batch, make sure at least one batch is available
                if self._current_batch is None:
                    if not self._buffer and not self._exhausted:
                        self._fill_buffer_single()
                    if self._buffer:
                        self._current_batch = self._buffer[0]
                return [self._current_batch] if self._current_batch is not None else []

            # Cap n to buffer_size to avoid infinite loops if caller requests
            # more batches than the buffer can physically hold.
            n = min(n, self.buffer_size)

            # Ensure buffer holds at least *n* items
            while len(self._buffer) < n and not self._exhausted:
                self._fill_buffer_single()

            # Return up to *n* items (may be fewer if exhausted)
            return list(self._buffer)[:n]
            
    def _fill_buffer(self):
        """Fill the buffer up to buffer_size"""
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            self._fill_buffer_single()
            
    def _fill_buffer_single(self):
        """Add a single batch to the buffer""" 
        # Initialize current iterator if needed
        if self._current_iterator is None:
            self._current_iterator = iter(self.dataloader)
 
        try:
            # Try to get batch from current iterator
            batch = next(self._current_iterator)
            self._buffer.append(batch)
        except StopIteration:
            # Current epoch exhausted - try to create iterator for next epoch
            if self._next_iterator is None:
                try:
                    self._next_iterator = iter(self.dataloader)
                except Exception:
                    # Can't create new iterator, we're done
                    self._exhausted = True
                    return
            
            # Try to get batch from next epoch's iterator
            try:
                batch = next(self._next_iterator)
                self._buffer.append(batch)
            except StopIteration:
                # Next iterator also exhausted, we're truly done
                self._exhausted = True
            
    def get_future_batches(self, start_offset: int, count: int) -> List[Any]:
        """
        Get future batches starting from start_offset positions ahead.
        This is used by async generation to get batches for future steps.
        
        Args:
            start_offset: How many batches ahead to start
            count: Number of batches to return
            
        Returns:
            List of batches (may be fewer than requested if not available)
        """
        with self._lock:
            # Ensure we have enough batches in buffer
            needed = start_offset + count
            while len(self._buffer) < needed and not self._exhausted:
                self._fill_buffer_single()
                
            # Extract the requested range
            result = []
            for i in range(start_offset, min(start_offset + count, len(self._buffer))):
                result.append(self._buffer[i])
                
            return result

    def __len__(self):
        """Return length of underlying dataloader if available"""
        return len(self.dataloader)
        
    @property
    def batch_size(self):
        """Return batch size of underlying dataloader"""
        return self.dataloader.batch_size
        
    @property
    def dataset(self):
        """Return dataset of underlying dataloader"""
        return self.dataloader.dataset

    @property
    def sampler(self):
        """Expose the sampler of the underlying dataloader.

        Having the `sampler` attribute available is required by several utilities in
        `transformers`/`accelerate` (e.g. `skip_first_batches`). This property simply
        forwards the call to the wrapped dataloader so that any logic depending on
        the sampler can function without modification.
        """
        return self.dataloader.sampler

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped dataloader when not found here.

        This makes the wrapper behave transparently like a regular `DataLoader` for
        most intents and purposes, exposing attributes such as `batch_sampler`,
        `drop_last`, etc., that might be accessed by external libraries.
        """
        return getattr(self.dataloader, name) 
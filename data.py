import tiktoken
class DataLoader:
    def __init__(self, data_path, batch_size, seq_len,process_rank,num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = self.load_data()

    def load_data(self):
        # Load your data here
        with open(self.data_path, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.current_index = self.seq_len * self.process_rank*self.batch_size
        return enc.encode(text)

    def get_batch(self):
        # Return a batch of data
        B,T = self.batch_size, self.seq_len
        buf = self.data[self.current_index:self.current_index+B*T+1]
        x = buf[:B*T].view(B,T)
        y = buf[1:B*T+1].view(B,T)
        self.current_index += B*T*self.num_processes
        if (self.current_index + B*T*self.num_processes+1) >= len(self.data):
            self.current_index = B*T*self.process_rank
        return x, y
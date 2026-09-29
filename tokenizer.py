import torch
from config import vocab_size, dialogue_id, dialogueStr

class BPETokenizer():

    def __init__(self, data):
        self.token_ids = list(data.encode("utf-8")) #list of ids of each byte        
        self.merges = {}
        self.idToToken = {}
        self.num_merges = vocab_size - 257
        self.replace_dialogueID(self.token_ids)
        self.bpe_merge()
        self.idsToTokens()
        self.tokenToId = {
            token: token_id
            for token_id, token in self.idToToken.items()
        }        
      
    def replace_dialogueID(self, byteIDs):
        dialogueIDs = list(dialogueStr.encode("utf-8"))

        i = 0
        dialogueSz = len(dialogueIDs)
        while i <= len(byteIDs) - dialogueSz:
            if (byteIDs[i:i+dialogueSz] == dialogueIDs): 
                byteIDs[i:i+dialogueSz] = [dialogue_id]
            i+=1

    def find_pair(self):
        counts = {}
        for pair in zip(self.token_ids, self.token_ids[1:]):
            if dialogue_id in pair:
                continue

            counts[pair] = counts.get(pair, 0) + 1

        most_frequent_pair = max(counts, key=counts.get)

        return most_frequent_pair

    def bpe_merge(self):

        for i in range(1, self.num_merges+1):
            pair = self.find_pair()
            tokenid = 256 + i
            self.merges[tokenid] = pair

            j = 0
            while(j < len(self.token_ids)- 1):

                if((self.token_ids[j] == pair[0]) and (self.token_ids[j+1] == pair[1])):
                    self.token_ids[j:j+2] = [tokenid]
                else:
                    j+=1
        
    def idsToTokens(self):

        for i in range(256):
            self.idToToken[i] = bytes([i]) #id 0-255 is its bytes 

        self.idToToken[dialogue_id] = b'<Dialogue>'

        for i in range (257, vocab_size): #257 or greater in merges
            pair = self.merges[i] # gives you a pair
            self.idToToken[i] = self.idToToken[pair[0]] + self.idToToken[pair[1]]

    def encode(self, text):
        token_ids = list(text.encode("utf-8"))
        self.replace_dialogueID(token_ids)

        for token_id, pair in self.merges.items():

            idx = 0
            while idx < len(token_ids) - 1:
                if pair == (token_ids[idx], token_ids[idx+1]):
                    token_ids[idx:idx+2] = [token_id]
                else:
                    idx+=1

        return torch.tensor(token_ids, dtype=torch.long)
    
    def getToken(self, token_id):
        return self.idToToken[token_id]

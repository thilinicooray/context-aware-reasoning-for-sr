import torch
import torchvision as tv
import json

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = []
        self.agent_label_list = []
        self.place_label_list = []
        self.max_role_count = 0
        self.question_words = {}
        self.vrole_question = {}

        self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                            'seller', 'boaters', 'blocker', 'farmer']

        # imag preprocess
        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomRotation(10),
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])


        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)
                self.verb2_role_dict[current_verb] = []

            roles = img['frames'][0].keys()
            has_agent = False
            has_place = False
            agent_role = None
            if 'place' in roles:
                has_place = True
            if 'agent' in roles:
                agent_role = 'agent'
                has_agent = True
            else:
                for role1 in roles:
                    if role1 in self.agent_roles[1:]:
                        agent_role = role1
                        has_agent = True
                        break

            for frame in img['frames']:
                for role,label in frame.items():
                    if role not in self.role_list:
                        self.role_list.append(role)
                    if role not in self.verb2_role_dict[current_verb]:
                        self.verb2_role_dict[current_verb].append(role)
                    if len(self.verb2_role_dict[current_verb]) > self.max_role_count:
                        self.max_role_count = len(self.verb2_role_dict[current_verb])
                    if label not in self.label_list:
                        self.label_list.append(label)
                    if label not in self.agent_label_list:
                        if has_agent and role == agent_role:
                            self.agent_label_list.append(label)
                    if label not in self.place_label_list:
                        if has_place and role == 'place':
                            self.place_label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t agent label count:', len(self.agent_label_list) ,
              '\n\t place label count:', len(self.place_label_list) ,
              '\n\t max role count:', self.max_role_count)


        verb2role_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb = []
            for role in current_role_list:
                role_id = self.role_list.index(role)
                role_verb.append(role_id)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_verb.append(len(self.role_list))

            verb2role_list.append(torch.tensor(role_verb))

        self.verb2role_list = torch.stack(verb2role_list)
        self.verb2role_encoding = self.get_verb2role_encoding()
        self.verb2role_oh_encoding = self.get_verb2role_oh_encoding()


    def get_verb2role_encoding(self):
        verb2role_embedding_list = []

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(1)


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(0)

            verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

        return verb2role_embedding_list

    def get_verb2role_oh_encoding(self):
        verb2role_oh_embedding_list = []

        role_oh = torch.eye(len(self.role_list)+1)

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(role_oh[self.role_list.index(role)])


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(role_oh[-1])

            verb2role_oh_embedding_list.append(torch.stack(role_embedding_verb, 0))

        return verb2role_oh_embedding_list

    def get_role_names(self, verb):
        current_role_list = self.verb2_role_dict[verb]

        role_verb = []
        for role in current_role_list:
            role_verb.append(self.role_corrected_dict[role])

        return role_verb

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def encode(self, item):
        verb = self.verb_list.index(item['verb'])
        labels = self.get_label_ids(item['verb'], item['frames'])

        return verb, labels

    def encode_with_impact(self, item):
        verb = self.verb_list.index(item['verb'])
        labels, impact = self.get_label_ids_with_impact(item['verb'], item['frames'])

        return verb, labels, impact

    def encode_agent(self, item):
        labels = self.get_agent_label_ids(item['verb'], item['frames'])

        return labels

    def encode_place(self, item):
        labels = self.get_place_label_ids(item['verb'], item['frames'])

        return labels

    def encode_verb(self, item):
        verb = self.verb_list.index(item['verb'])

        return verb

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

    def get_role_ids_batch(self, verbs):
        role_batch_list = []
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_agent_place_ids_batch(self, batch_size):
        role_batch_list = []
        agent_place = torch.tensor([self.role_list.index('agent'),self.role_list.index('place')])
        for i in range(batch_size):
            role_batch_list.append(agent_place)

        return torch.stack(role_batch_list,0)

    def get_label_ids(self, verb, frames):
        all_frame_id_list = []
        roles = self.verb2_role_dict[verb]
        for frame in frames:
            label_id_list = []

            for role in roles:
                label = frame[role]
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('UNK')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(len(self.label_list))

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels

    def get_agent_label_ids(self, verb, frames):
        agent_id_list = []
        roles = self.verb2_role_dict[verb]

        has_agent = False
        if 'agent' in roles:
            agent_role = 'agent'
            has_agent = True
        else:
            for role1 in roles:
                if role1 in self.agent_roles[1:]:
                    agent_role = role1
                    has_agent = True
                    break

        for frame in frames:
            if has_agent:
                agent = frame[agent_role]
                if agent in self.agent_label_list:
                    label_id = self.agent_label_list.index(agent)
                else:
                    label_id = self.agent_label_list.index('UNK')
                agent_id_list.append(label_id)
            else:
                agent_id_list.append(len(self.agent_label_list))

        labels = torch.tensor(agent_id_list)

        return labels

    def get_place_label_ids(self, verb, frames):
        place_id_list = []
        roles = self.verb2_role_dict[verb]

        has_place = False

        if 'place' in roles:
            has_place = True

        for frame in frames:

            if has_place:
                place = frame['place']
                if place in self.place_label_list:
                    label_id = self.place_label_list.index(place)
                else:
                    label_id = self.place_label_list.index('UNK')

                place_id_list.append(label_id)

            else:
                place_id_list.append(len(self.place_label_list))

        labels = torch.tensor(place_id_list)

        return labels

    def get_adj_matrix(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(encoding.clone().detach(),0)
            role_count = self.get_role_count(id)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_adj_matrix_noself(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(encoding.clone().detach(), 0)
            role_count = self.get_role_count(id)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx1 in range(0,role_count):
                adj[idx1][idx1] = 0
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_verb2role_encoing_batch(self, verb_ids):
        matrix_list = []

        for id in verb_ids:
            encoding = self.verb2role_encoding[id]
            matrix_list.append(encoding)

        encoding_all = torch.stack(matrix_list).type(torch.FloatTensor)

        return encoding_all





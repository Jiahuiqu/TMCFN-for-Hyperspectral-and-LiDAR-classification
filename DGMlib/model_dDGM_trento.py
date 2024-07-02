from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGraphConv, GCNConv, GATConv, SAGEConv
from DGMlib.layers_dense import *


class DGM_Model(torch.nn.Module):
    def __init__(self, hparams):
        super(DGM_Model, self).__init__()

        self.hparams = hparams
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k

        self.graph_f = ModuleList()
        self.node_g_1 = ModuleList()
        self.node_g_2 = ModuleList()
        self.node_g_3 = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            if len(dgm_l) > 0:
                if 'ffun' not in hparams or hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                if hparams.ffun == 'sage':
                    self.graph_f.append(DGM_d(SAGEConv(dgm_l[0], dgm_l[-1]), k=hparams.k, distance=hparams.distance))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l), k=hparams.k, distance=hparams.distance))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0), k=hparams.k, distance=hparams.distance))

            else:
                self.graph_f.append(Identity())

            if len(conv_l) > 0:
                if hparams.gfun == 'edgeconv':
                    conv_l = conv_l.copy()
                    conv_l[0] = conv_l[0] * 2
                    self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
                if hparams.gfun == 'gcn':
                    self.node_g_1.append(GCNConv(conv_l[0], conv_l[1]))
                    self.node_g_2.append(GCNConv(conv_l[0], conv_l[1]))
                    self.node_g_3.append(GCNConv(conv_l[0], conv_l[1]))
                if hparams.gfun == 'sage':
                    self.node_g_1.append(SAGEConv(conv_l[0], conv_l[1]))
                    self.node_g_2.append(SAGEConv(conv_l[0], conv_l[1]))
                    self.node_g_3.append(SAGEConv(conv_l[0], conv_l[1]))
                if hparams.gfun == 'gat':
                    self.node_g_1.append(GATConv(conv_l[0], conv_l[1]))
                    self.node_g_2.append(GATConv(conv_l[0], conv_l[1]))
                    self.node_g_3.append(GATConv(conv_l[0], conv_l[1]))
            else:
                self.node_g_1.append(Identity())
                self.node_g_2.append(Identity())
                self.node_g_3.append(Identity())

    def forward(self, x1, x2, x3, init_edges=None, warm_up=False):

        lprobslist = []
        self.edges = init_edges.clone()
        for f, g1, g2, g3 in zip(self.graph_f, self.node_g_1, self.node_g_2, self.node_g_3):
            b, n, d = x1.shape
            x1 = torch.nn.functional.relu(g1(torch.dropout(x1.view(-1, d), 0.3, train=self.training), self.edges)).view(
                b, n, -1)
            x2 = torch.nn.functional.relu(g2(torch.dropout(x2.view(-1, d), 0.3, train=self.training), self.edges)).view(
                b, n, -1)
            x3 = torch.nn.functional.relu(g3(torch.dropout(x3.view(-1, d), 0.3, train=self.training), self.edges)).view(
                b, n, -1)

            # x1 = g1(x1.view(-1, d), self.edges).view(b, n, -1) # torch.Size([1, 2708, 32])
            # x2 = g2(x2.view(-1, d), self.edges).view(b, n, -1)  # torch.Size([1, 2708, 32])
            # x3 = g3(x3.view(-1, d), self.edges).view(b, n, -1)  # torch.Size([1, 2708, 32])

            graph_x = torch.cat([x1.detach(), x2.detach(), x3.detach()], -1)

            graph_x, edges, lprobs = f(graph_x, self.edges, None)
            if lprobs is not None:
                lprobslist.append(lprobs)
            if (warm_up == True):
                self.edges = init_edges
            else:
                self.edges = edges

        return x1, x2, x3, self.edges, torch.stack(lprobslist, -1) if len(lprobslist) > 0 else None

    def training_step(self, train_batch, batch_idx):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer.zero_grad()
        X, y, mask, edges = train_batch
        edges = edges[0]

        assert (X.shape[0] == 1)  # only works in transductive setting
        mask = mask[0]
        ## logprobs
        pred, logprobs = self(X, edges)

        train_pred = pred[:, mask.to(torch.bool), :]
        train_lab = y[:, mask.to(torch.bool), :]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred, train_lab)
        loss.backward()

        correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()

        if logprobs is not None:
            corr_pred = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().detach()
            wron_pred = (1 - corr_pred)

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * 0.5

            point_w = (self.avg_accuracy - corr_pred)
            graph_loss = point_w * logprobs[:, mask.to(torch.bool), :].exp().mean([-1, -2])

            graph_loss = graph_loss.mean()
            graph_loss.backward()

            self.log('train_graph_loss', graph_loss.detach().cpu())
            if (self.debug):
                self.point_w = point_w.detach().cpu()

            self.avg_accuracy = self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred

        optimizer.step()

        self.log('train_acc', correct_t)
        self.log('train_loss', loss.detach().cpu())

    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]

        assert (X.shape[0] == 1)  # only works in transductive setting
        mask = mask[0]
        pred, logprobs = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edges)
            pred += pred_.softmax(-1)
        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)
        self.log('test_loss', loss.detach().cpu())
        #         self.log('test_graph_loss', loss.detach().cpu())
        self.log('test_acc', 100 * correct_t)

    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        edges = edges[0]

        assert (X.shape[0] == 1)  # only works in transductive setting
        mask = mask[0]

        pred, logprobs = self(X, edges)
        pred = pred.softmax(-1)
        for i in range(1, self.hparams.test_eval):
            pred_, logprobs = self(X, edges)
            pred += pred_.softmax(-1)

        test_pred = pred[:, mask.to(torch.bool), :]
        test_lab = y[:, mask.to(torch.bool), :]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)

        self.log('val_loss', loss.detach())
        self.log('val_acc', 100 * correct_t)





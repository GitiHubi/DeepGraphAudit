# init client argument parser
parser = argparse.ArgumentParser(description='deepApple Experiments')

# general experiment parameter
parser.add_argument('-exp_timestamp', help='', nargs='?', type=str,  default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('-data_dir', help='', nargs='?', type=str, default='./100_datasets')
parser.add_argument('-base_dir', help='', nargs='?', type=str, default='./200_experiments')
parser.add_argument('-exp_postfix', type=str, default='poc', help='postfix of experimental runs.')

# data parameter
parser.add_argument('-dataset', help='', nargs='?', type=str,  default='ernstyoung')
parser.add_argument('-sample_eval', help='', nargs='?', type=str, default='False')
parser.add_argument('-sample_size', help='', nargs='?', type=int, default=10000)

# model architecture parameter
parser.add_argument('-seed', type=int, default=1111, help='seed value for deterministic results.')
parser.add_argument('-encoder_dim', help='', nargs='+', default=[64, 32, 16, 8, 4, 2]) # [8, 4, 2], [5096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
parser.add_argument('-bottleneck', help='', nargs='?', type=str, default='linear')
parser.add_argument('-decoder_dim', help='', nargs='+', default=[2, 4, 8, 16, 32, 64]) # [2, 4, 8], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096]

# model training parameter
parser.add_argument('-iterations', type=int, default=100001, help='the number of training iterations.')
parser.add_argument('-eval_iteration', type=int, default=10000, help='the eval training iteration.')
parser.add_argument('-batch_size', type=int, default=64, help='the batch size.')
parser.add_argument('-learning_rate', type=float, default=0.0001, help='the learning rate.')
parser.add_argument('-device', type=str, default='cpu', help='the compute device.')

# ignore potential warnings - mostly due to package updates
warnings.filterwarnings('ignore')

# parse client arguments
parameter = vars(parser.parse_args())

# init new handlers
uha = UtilsHandler.UtilsHandler()
dha = DataHandler.DataHandler()
vha = VisualisationHandler.VisualizationHandler()

# parse boolean args as boolean
parameter['sample_eval'] = uha.str2bool(parameter['sample_eval'])

# set deterministic seeds of the client training experiments
np.random.seed(int(parameter['seed']))  # set numpy seed
th.manual_seed(int(parameter['seed']))  # set pytorch seed CPU
th.cuda.manual_seed(int(parameter['seed']))  # set pytorch seed GPU

# parse string args as int
parameter['encoder_dim'] = [int(ele) for ele in parameter['encoder_dim']]
parameter['decoder_dim'] = [int(ele) for ele in parameter['decoder_dim']]

# create experiment directory
parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_baseline')

# load the EY training data
processed_entries, aggregated_entries, prepared_entries, entries_statistics = dha.get_ernstyoung_data_range(parameter=parameter)

# log aggregated categorical entries
file_name = '{}_sampled_processed_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
processed_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

# log aggregated categorical entries
file_name = '{}_sampled_aggregated_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

# log aggregated categorical entries
file_name = '{}_sampled_prepared_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
prepared_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

# convert the EY training data to pytorch tensor
prepared_tensor_entries = AccountingDataset(dataset=prepared_entries.values.astype(np.float32))

#### start training routine

# init the EY data loader
train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=True)

# update encoder and decoder network dimensions
parameter['encoder_dim'].insert(0, prepared_entries.shape[1])
parameter['decoder_dim'].insert(len(parameter['encoder_dim']), prepared_entries.shape[1])

# init the autoencoder model
model = Autoencoder.Autoencoder(
    encoder_layer=parameter['encoder_dim'],
    bottleneck=parameter['bottleneck'],
    decoder_layer=parameter['decoder_dim'],
).to(parameter['device'])

# init aggregated categorical autoencoder loss
rec_criterion = th.nn.MSELoss().to(parameter['device'])

# init training optimizer
optimizer = th.optim.Adam(model.parameters(), lr=parameter['learning_rate'])

# set model in train mode
model.train()

# init the training loss
average_train_loss = 0.0

# push aggregated losses to compute device
rec_criterion_cat = rec_criterion.to(parameter['device'])

# init and wrap range of training iterations
training_iterations = tqdm(range(0, parameter['iterations']))

# case: mini-batches are still available
for i in training_iterations:

    # get next training batch
    inputs, targets = next(iter(train_loader))

    # push the inputs and targets to compute device
    inputs, targets = inputs.to(parameter['device']), targets.to(parameter['device'])

    # reset optimizer gradients
    optimizer.zero_grad()

    # run model forward pass
    latent, reconstructions = model(inputs)

    # compute and add categorical reconstruction loss
    train_batch_loss = rec_criterion_cat(input=reconstructions, target=inputs)

    # compute and collect average reconstruction loss
    average_train_loss += train_batch_loss.cpu().detach().item()

    # log training progress
    now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
    training_iterations.set_description(
        (
            '[INFO {}] DeepAppleGraph :: iteration: {}, train-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(train_batch_loss.cpu().detach().item(), 6)))
        )
    )

    # run backward pass
    train_batch_loss.backward()

    # run optimizer step
    optimizer.step()

    # case: eval iteration
    if i % parameter['eval_iteration'] == 0:

        # save client model checkpoint
        file_name = '{}_ae_baseline_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
        uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

#### start evaluation routine

# init the EY data loader
eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=False)

# set model in evaluation mode
model.eval()

# init validation embeddings
valid_embeddings = []

# init number of batch iterations
batch_count = 0

# case: mini-batches are still available
for batch, _ in eval_loader:

    # push the inputs to compute device
    batch = batch.to(parameter['device'])

    # run model forward pass
    valid_embeddings_batch, _ = model(batch)

    # case: initial batch
    if batch_count == 0:

        # collect validation embeddings
        valid_embeddings = valid_embeddings_batch.cpu().detach().numpy()

    # case: non-initial batch
    else:

        # collect validation embeddings
        valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

    # increase batch iteration
    batch_count += 1

# update journal entries with embedding
aggregated_entries['z1'] = valid_embeddings[:, 0]
aggregated_entries['z2'] = valid_embeddings[:, 1]

# log aggregated categorical entries
file_name = '{}_sampled_aggregated_entries_embedded_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

# set visualization handler directory
vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

# visualize learned embeddings
filename = '{}_je_embedding_distribution_it_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
title = 'AE Baseline - JE Embedding Distribution\niterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
vha.plot_embeddings_2d(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

# visualize learned embeddings
filename = '{}_je_embedding_distribution_it_{}_at_PREPARER.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
title = 'AE Baseline - JE Embedding Distribution\nattribute: Y_PREPARER_ID, iterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
vha.plot_embeddings_2d_attribute(data=aggregated_entries, attribute='Y_PREPARER_ID', z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

# visualize learned embeddings
filename = '{}_je_embedding_distribution_it_{}_at_SOURCE.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
title = 'AE Baseline - JE Embedding Distribution\nattribute: Y_SOURCE, iterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
vha.plot_embeddings_2d_attribute(data=aggregated_entries, attribute='Y_SOURCE', z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

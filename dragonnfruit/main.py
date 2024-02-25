import argparse
import pandas as pd
import numpy as np
import pickle5
import torch

from dragonnfruit.io import GenomewideGenerator
from dragonnfruit.io import LocusGenerator
from dragonnfruit.io import GenomewideWeightedGenerator

from dragonnfruit.models import CellStateController
from dragonnfruit.models import DynamicBPNet
from dragonnfruit.models import DragoNNFruit

from chrombpnet.evaluation.make_bigwigs.predict_to_bigwig import load_model_wrapper

def main():
    parser = argparse.ArgumentParser(description="A main function to set up dragonnfruit run")

    # Necessary parameters
    parser.add_argument("neighbors", help="An numpy file path storing the top N neighbors of each cell", type=str)
    parser.add_argument("cell_states", help="A tsv file path storing the cell embeddings", type=str)
    parser.add_argument("read_depths", help="An numpy file path storing the read depths of cells", type=str)
    parser.add_argument("signal", help="A pickle file path storing the sparse matrices of single cell ATAC seq signal", type=str)
    parser.add_argument("sequences", help="A pickle file path storing the sequence data for each chromosome", type=str)
    parser.add_argument("peaks", help="A bed file path storing the peaks", type=str)
    parser.add_argument("bias_model", help="A file path storing the trained bias model", type=str)

    # Add optional parameters
    parser.add_argument("-t", "--training_set", help="The chromosomes in the training set", type=str, default=["chr11"])
    parser.add_argument("-v", "--validation_set", help="The chromosomes in the validation set", type=str, default=["chr15"])


    args = parser.parse_args()

    # Load Neighbors
    neighbors = np.load(args.neighbors)

    # Load cell state files
    cell_states_df = pd.read_csv(args.cell_states, sep='\t')
    cell_states = cell_states_df.to_numpy()
    cell_states = (cell_states - cell_states.mean(axis=0, keepdims=True)) / cell_states.std(axis=0, keepdims=True)
    cell_states = cell_states.astype('float32')

    # Load read_depths
    read_depths = np.load(args.read_depths)
    read_depths = read_depths[neighbors].sum(axis=1)
    read_depths = np.log2(read_depths + 1).reshape(-1, 1)

    # Load signal
    with open(args.signal, 'rb') as file:
        signals = pickle5.load(file)
    
    # Get the used chromosomes
    chromosomes = args.training_set + args.validation_set
    signals = {k: signals[k] for k in chromosomes}

    # Load the sequence file
    with open(args.sequences, 'rb') as file:
        sequences = pickle5.load(file)

    # Define peak file
    peak_file = args.peaks

    # Training set data generator
    sequences_train = {key:sequences[key] for key in args.training_set}
    signals_train = {key:signals[key] for key in args.training_set}
    X = torch.utils.data.DataLoader(
        GenomewideGenerator(
            sequence=sequences_train,
            signal=signals_train,
            neighbors=neighbors,
            cell_states=cell_states,
            read_depths=read_depths,
            trimming=(2114 - 1000) // 2,
            window=2114,
            chroms=args.training_set,
            random_state=None),
        pin_memory=True,
        num_workers=1,
        worker_init_fn=lambda x: np.random.seed(x),
        batch_size=128)

    sequences_validate = {key:sequences[key] for key in args.validation_set}
    signals_validate = {key:signals[key] for key in args.validation_set}
    X_valid = LocusGenerator(
        sequence=sequences_validate,
        signal=signals_validate,
        loci_file=peak_file,
        neighbors=neighbors,
        cell_states=cell_states,
        read_depths=read_depths,
        trimming=(2114 - 1000) // 2,
        window=2114,
        chroms=args.validation_set,
        random_state=0)
    
    # Load the bias model
    bias_model = load_model_wrapper(args.bias_model)

    controller = CellStateController(n_inputs=cell_states.shape[-1], n_nodes=1024, 
	n_layers=1, n_outputs=128).cuda()

    accessibility_model = DynamicBPNet(n_filters=256, n_layers=8,
	trimming=(2114 - 1000) // 2, controller=controller).cuda()

    model = DragoNNFruit(bias_model, accessibility_model, "dragonnfruit").cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.fit(X, X_valid, optimizer, n_validation_samples=5000, max_epochs=100, 
	validation_iter=250, batch_size=64)


if __name__ == "__main__":
    main()
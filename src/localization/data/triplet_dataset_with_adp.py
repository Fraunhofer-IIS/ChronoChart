#%%
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
from scipy.sparse.csgraph import dijkstra
import json
import os
import torch
from src.localization.data.triplet_dataset import BaseTripletDataset


# Antenna definitions
ASSIGNMENTS = [
	[0, 13, 31, 29, 3, 7, 1, 12 ],
	[30, 26, 21, 25, 24, 8, 22, 15],
	[28, 5, 10, 14, 6, 2, 16, 18],
	[19, 4, 23, 17, 20, 11, 9, 27]
]

ANTENNACOUNT = np.sum([len(antennaArray) for antennaArray in ASSIGNMENTS])

def load_calibrate_timedomain(path, offset_path):

	offsets = None
	with open(offset_path, "r") as offsetfile:
		offsets = json.load(offsetfile)
	
	def record_parse_function(proto):
		record = tf.io.parse_single_example(
			proto,
			{
				"csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
			},
		)

		csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (ANTENNACOUNT, 1024, 2))
		csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
		csi = tf.signal.fftshift(csi, axes=1)

		position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
		time = tf.ensure_shape(record["time"], ())

		return csi, position[:2], time

	def apply_calibration(csi, pos, time):
		sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
		cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
		csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

		return csi, pos, time

	def csi_time_domain(csi, pos, time):
		csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes=1)),axes=1)

		return csi, pos, time

	def cut_out_taps(tap_start, tap_stop):
		def cut_out_taps_func(csi, pos, time):
			return csi[:,tap_start:tap_stop], pos, time

		return cut_out_taps_func

	def order_by_antenna_assignments(csi, pos, time):
		csi = tf.stack([tf.gather(csi, antenna_inidces) for antenna_inidces in ASSIGNMENTS])
		return csi, pos, time
	
	dataset = tf.data.TFRecordDataset(path)
	
	dataset = dataset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(apply_calibration, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)

	return dataset
def get_csi_time_ref():
    inputpaths = [
        {
            "tfrecords" : "dichasus-cf02.tfrecords",
            "offsets" : "reftx-offsets-dichasus-cf02.json"
        },
        {
            "tfrecords" : "dichasus-cf03.tfrecords",
            "offsets" : "reftx-offsets-dichasus-cf03.json"
        },
        {
            #"tfrecords" : "dichasus-cf07.tfrecords",
            #"offsets" : "reftx-offsets-dichasus-cf07.json"
            "tfrecords" : "dichasus-cf04.tfrecords",
            "offsets" : "reftx-offsets-dichasus-cf04.json"
        },
        


    ]


    groundtruth_positions = []
    csi_time_domain = []
    timestamps= []
    from src.localization.config.paths import PathConfig
    pc = PathConfig()
    for path in inputpaths:

        timestamps_1 = []

        full_dataset = load_calibrate_timedomain(pc.dichasus / path["tfrecords"],pc.dichasus / path["offsets"])
        training_set = full_dataset#.shard(4, 0)
        for csi, pos, time in training_set.batch(4000):

            csi_time_domain.append(csi.numpy())
            groundtruth_positions.append(pos.numpy())
            timestamps_1.append(time.numpy())

        timestamps_1 = np.concatenate(timestamps_1)
        #timestamps_1 =(1- (timestamps_1-timestamps_1.min()) / (timestamps_1.max() - timestamps_1.min()) )
        print(len(timestamps_1))
        timestamps.append(timestamps_1)

        print(len(timestamps))


    timestamps_1=None
    csi = None

    csi_time_domain = np.concatenate(csi_time_domain)
    groundtruth_positions = np.concatenate(groundtruth_positions)
    timestamps = np.concatenate(timestamps)

    return csi_time_domain,groundtruth_positions,timestamps




def get_adp(csi_time_domain):
    adp_dissimilarity_matrix = np.zeros((csi_time_domain.shape[0], csi_time_domain.shape[0]),dtype=np.float32)
    def adp_dissimilarities_worker(todo_queue, output_queue):
        def adp_dissimilarities(index):
            # h has shape (arrays, antennas, taps), w has shape (datapoints, arrays, antennas, taps)
            h = csi_time_domain[index,:,:,:]
            w = csi_time_domain[index:,:,:,:]
                
            dotproducts = np.abs(np.einsum("bmt,lbmt->lbt", np.conj(h), w, optimize = "optimal"))**2
            norms = np.real(np.einsum("bmt,bmt->bt", h, np.conj(h), optimize = "optimal") * np.einsum("lbmt,lbmt->lbt", w, np.conj(w), optimize = "optimal"))
            
            return np.sum(1 - dotproducts / norms, axis = (1, 2))

        while True:
            index = todo_queue.get()

            if index == -1:
                output_queue.put((-1, None))
                break
            
            output_queue.put((index, adp_dissimilarities(index)))


    with tqdm(total = csi_time_domain.shape[0]**2) as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(csi_time_domain.shape[0]):
            todo_queue.put(i)
        
        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target = adp_dissimilarities_worker, args = (todo_queue, output_queue))
            p.start()

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            i, d = output_queue.get()

            if i == -1:
                finished_processes = finished_processes + 1
            else:
                adp_dissimilarity_matrix[i,i:] = d
                adp_dissimilarity_matrix[i:,i] = d
                pbar.update(2 * len(d) - 1)

        
    



    return adp_dissimilarity_matrix

def get_geodistance(adp_dissimilarity_matrix,timestamp_dissimilarity_matrix,n_neighbors=20):
    if isinstance(adp_dissimilarity_matrix,torch.Tensor):
         adp_dissimilarity_matrix = adp_dissimilarity_matrix.numpy()
    
    if isinstance(timestamp_dissimilarity_matrix,torch.Tensor):
         timestamp_dissimilarity_matrix = timestamp_dissimilarity_matrix.numpy()
    

    TIME_THRESHOLD = 2
    small_time_dissimilarity_indices = np.logical_and(timestamp_dissimilarity_matrix < TIME_THRESHOLD, timestamp_dissimilarity_matrix > 0)
    small_time_dissimilarities = timestamp_dissimilarity_matrix[small_time_dissimilarity_indices]
    small_adp_dissimilarities = adp_dissimilarity_matrix[small_time_dissimilarity_indices]

    n_bins = 1500

    fig, ax1 = plt.subplots()
    occurences, edges, patches = ax1.hist(small_adp_dissimilarities / small_time_dissimilarities, range = (0, 50), bins = n_bins)
    ax1.set_xlabel(r"$D_\mathrm{APDP} / D_\mathrm{time}$")
    ax1.set_ylabel("Number of Occurences")
    bin_centers = edges[:-1] + np.diff(edges) / 2.
    max_bin = np.argmax(occurences)
    min_threshold = np.quantile(occurences[:max_bin], 0.5)

    for threshold_bin in range(max_bin - 1, -1, -1):
        if occurences[threshold_bin] < min_threshold:
            break

    scaling_factor = bin_centers[threshold_bin]

    print("scaling factor",scaling_factor)


    dissimilarity_matrix_fused =np.minimum(adp_dissimilarity_matrix, timestamp_dissimilarity_matrix * scaling_factor)
        

    nbrs_alg = NearestNeighbors(n_neighbors = n_neighbors, metric="precomputed", n_jobs = -1)
    nbrs = nbrs_alg.fit(dissimilarity_matrix_fused)
    nbg = kneighbors_graph(nbrs, n_neighbors, metric = "precomputed", mode="distance")

    dissimilarity_matrix_geodesic = np.zeros((nbg.shape[0], nbg.shape[1]), dtype = np.float32)

    def shortest_path_worker(todo_queue, output_queue):
        while True:
            index = todo_queue.get()

            if index == -1:
                output_queue.put((-1, None))
                break

            d = dijkstra(nbg, directed=False, indices=index)
            output_queue.put((index, d))

    with tqdm(total = nbg.shape[0]**2) as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(nbg.shape[0]):
            todo_queue.put(i)
        
        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target = shortest_path_worker, args = (todo_queue, output_queue))
            p.start()

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            i, d = output_queue.get()

            if i == -1:
                finished_processes = finished_processes + 1
            else:
                dissimilarity_matrix_geodesic[i,:] = d
                pbar.update(len(d))
    

    return dissimilarity_matrix_geodesic

def isotonic_decreasing_torch(x):
    y = x.clone().float()
    n = y.numel()
    weights = torch.ones(n, dtype=y.dtype, device=y.device)

    i = 0
    while i < n - 1:
        if y[i] < y[i + 1]:
            total_weight = weights[i] + weights[i + 1]
            avg = (weights[i] * y[i] + weights[i + 1] * y[i + 1]) / total_weight

            y[i] = y[i + 1] = avg
            weights[i] = weights[i + 1] = total_weight

            j = i
            while j > 0 and y[j - 1] < y[j]:
                total_weight = weights[j - 1] + weights[j]
                avg = (weights[j - 1] * y[j - 1] + weights[j] * y[j]) / total_weight
                y[j - 1] = y[j] = avg
                weights[j - 1] = weights[j] = total_weight
                j -= 1
        else:
            i += 1

    return y


#%%
class TripletDatasetWithADP(BaseTripletDataset):
    
    def __init__(self, K, temporal_window_seconds=None):
        super().__init__(None, None, None, K, temporal_window_seconds)
        from src.localization.config.paths import PathConfig
        self.path_config = PathConfig()
        try:
            self._load()
        except Exception as e:
            print(e)
            self._save()
            self._load()

        print("data loaded")
        self._preprocess()   
        print("data preprocessed")     

        
    def __len__(self):
        return len(self.positive_indexes) #len(self.keep_idxs) # Or however many pairs you want
    
    def _save(self):
        device = torch.device('cpu')
        dtype = torch.float32
        mmap = True
        ptdir = self.path_config.pt_dir
        ptdir.mkdir(parents=True, exist_ok=True)

        csi_time_domain,groundtruth_positions,timestamps = get_csi_time_ref()
        self.csi = csi_time_domain
        self.timestamps = torch.from_numpy(timestamps).to(dtype=dtype, device=device)
        self.reference = torch.from_numpy(groundtruth_positions).to(dtype=dtype, device=device)

        np.save(ptdir/'csi.npy',self.csi)
        torch.save(self.timestamps,ptdir/'timestamps.pt')
        torch.save(self.reference,ptdir/'reference.pt')

                
    def _load(self):
        device = torch.device('cpu')
        dtype = torch.float32
        mmap = False
        ptdir = self.path_config.pt_dir
        print("loading data from", ptdir)

        self.csi = np.load(ptdir/'csi.npy')#,mmap_mode='r+')
        self.timestamps = torch.load(ptdir/'timestamps.pt',map_location=device,mmap=mmap)
        self.reference = torch.load(ptdir/'reference.pt',map_location=device,mmap=mmap)


    def __len__(self):
        return self._len__()
        
    def __getitem__(self, index):
         return self._getitem__(index)
# %%

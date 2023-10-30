# This file is from Ufuk
# https://github.com/ufuk-cakir/MEGS

import requests
import os
import h5py
import numpy as np


class illustrisAPI:
    DATAPATH = "./tempdata"
    URL = "http://www.tng-project.org/api/"

    def __init__(
        self,
        api_key,
        particle_type="stars",
        simulation="TNG100-1",
        snapshot=99,
    ):
        """Illustris API class.

        Class to load data from the Illustris API.

        Parameters
        ----------
        api_key : str
            API key for the Illustris API.
        particle_type : str
            Particle type to load. Default is "stars".
        simulation : str
            Simulation to load from. Default is "TNG100-1".
        snapshot : int
            Snapshot to load from. Default is 99.
        """

        self.headers = {"api-key": api_key}
        self.particle_type = particle_type
        self.snapshot = snapshot
        self.simulation = simulation
        self.baseURL = f"{self.URL}{self.simulation}/snapshots/{self.snapshot}"

    def get(self, path, params=None, name=None):
        """Get data from the Illustris API.

        Parameters
        ----------
        path : str
            Path to load from.
        params : dict
            Parameters to pass to the API.
        name : str
            Name to save the file as. If None, the name will be taken from the content-disposition header.

        Returns
        -------
        r : requests object
            The requests object.

        """

        os.makedirs(self.DATAPATH, exist_ok=True)
        r = requests.get(path, params=params, headers=self.headers)
        # raise exception if response code is not HTTP SUCCESS (200)
        r.raise_for_status()
        if r.headers["content-type"] == "application/json":
            return r.json()  # parse json responses automatically
        if "content-disposition" in r.headers:
            filename = (
                r.headers["content-disposition"].split("filename=")[1]
                if name is None
                else name
            )
            with open(f"{self.DATAPATH}/{filename}.hdf5", "wb") as f:
                f.write(r.content)
            return filename  # return the filename string
        return r

    def get_subhalo(self, id):
        """Get subhalo data from the Illustris API.

        Returns the subhalo data for the given subhalo ID.

        Parameters
        ----------
        id : int
            Subhalo ID to load.

        Returns
        -------
        r : dict
            The subhalo data.

        """

        return self.get(f"{self.baseURL}/subhalos/{id}")

    def load_hdf5(self, filename):
        """Load HDF5 file.

        Loads the HDF5 file with the given filename.

        Parameters
        ----------
        filename : str
            Filename to load.

        Returns
        -------
        returndict : dict
            Dictionary containing the data from the HDF5 file.
        """
        # Check if filename ends with .hdf5
        if filename.endswith(".hdf5"):
            filename = filename[:-5]

        returndict = dict()

        with h5py.File(f"{self.DATAPATH}/{filename}.hdf5", "r") as f:
            for type in f.keys():
                if type == "Header":
                    continue
                if type.startswith("PartType"):
                    for fields in f[type].keys():
                        returndict[fields] = f[type][fields][()]

        return returndict

    def filter_masses(self, log_M_min=11.0, log_M_max=12.0):
        '''Filter subhalos by mass.
        
        Returns the IDs of subhalos with masses between log_M_min and log_M_max.
        
        Parameters
        ----------
        log_M_min : float
            Minimum mass in log10(M_sun/h).
        log_M_max : float
            Maximum mass in log10(M_sun/h).

        Returns
        -------
        ids : list
            List of subhalo IDs.
        '''
        # Convert to physical units
        mass_min = 10**log_M_min / 1e10 * 0.704
        mass_max = 10**log_M_max / 1e10 * 0.704
        # Create search query
        search_query = "?mass__gt=" + str(mass_min) + "&mass__lt=" + str(mass_max)
        subhalos = self.get(f"{self.baseURL}/subhalos/{search_query}")
        # Get number of subhalos
        count = subhalos["count"]
        # Query all subhalos, setting the limit to the number of subhalos,
        # because default limit is 100 
        search_query = search_query + "&limit=" + str(count)
        subhalos = self.get(f"{self.baseURL}/subhalos/{search_query}")
        # return IDS 
        ids = [subhalo["id"] for subhalo in subhalos["results"]]
        return ids

    def get_particle_data(self, id, fields):
        """Get particle data from the Illustris API.

        Returns the particle data for the given subhalo ID.

        Parameters
        ----------
        id : int
            Subhalo ID to load.
        fields : str or list
            Fields to load. If a string, the fields should be comma-separated.

        Returns
        -------
        data : dict
            Dictionary containing the particle data in the given fields.
        """
        # Get fields in the right format
        if isinstance(fields, str):
            fields = [fields]
        fields = ",".join(fields)

        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?{self.particle_type}={fields}"
        self.get(url, name="cutout")
        data = self.load_hdf5("cutout")

        return data


if __name__ == '__main__':
    # remember to delete key
    key = ""
    
    tuple_list = [("TNG50-1", 99), ("TNG100-1", 99), ("illustris-1", 135)]
    for t in tuple_list:
        a = illustrisAPI(api_key = key, particle_type="stars", simulation = t[0], snapshot = t[1])
        ids = a.filter_masses(log_M_min = 9.0, log_M_max = 13)
        print(len(ids))
        np.save('./api_filter_mass_id/' + t[0][: -2] + '_1_snap' + str(t[1]) + '.npy', ids)


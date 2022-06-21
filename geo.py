#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
import sklearn.cluster
import numpy as np

chosen_region="STC"

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovanie dataframe do geopandas.GeoDataFrame s Krovakovym zobrazenim (EPSG:5514)
    
    df  : dataframe ktory chceme prekonvertovat

    """
    dfc = df.loc[~np.isnan(df["d"]) & ~np.isnan(df["e"])]
    gdf = geopandas.GeoDataFrame(dfc, geometry=geopandas.points_from_xy(dfc["d"], dfc["e"]),  crs="EPSG:5514")
    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """ Vykresleni grafu so siestimi podgrafmi podla lokality nehody 
     (dialnica vs prva trida) pre roky 2018-2020 
     
     gdf            : GeoDataFrame so zaznamami nehod, ktore chceme zobrazit
     fig_location   : ak nie je None a nie je prazdny string, tak ulozi obrazok na tuto cestu
     show_figure    : uzivatelska volba pre vykreslenie grafov
     """
    gdf["p2a"] = pd.to_datetime(gdf["p2a"])
     # pouzijeme Mercatorovo zobrazenie, pri Krovakovom podkladova mapa otocena o par stupnov v smere hodinovych ruciciek
    gdf = gdf.loc[gdf["region"] == chosen_region].to_crs("EPSG:3857")
    bounds = gdf.total_bounds
    minx = bounds[0]
    maxx = bounds[2]
    d_2018 = gdf.loc[(gdf["p2a"].dt.year == 2018) & (gdf["p36"] == 0)]
    d_2019 = gdf.loc[(gdf["p2a"].dt.year == 2019) & (gdf["p36"] == 0)]
    d_2020 = gdf.loc[(gdf["p2a"].dt.year == 2020) & (gdf["p36"] == 0)]
    c1_2018 = gdf.loc[(gdf["p2a"].dt.year == 2018) & (gdf["p36"] == 1)]
    c1_2019 = gdf.loc[(gdf["p2a"].dt.year == 2019) & (gdf["p36"] == 1)]
    c1_2020 = gdf.loc[(gdf["p2a"].dt.year == 2020) & (gdf["p36"] == 1)]
    fig, axs = plt.subplots(3, 2, figsize=(10,14))
    for i, ax in enumerate(axs.flatten()):
        if i % 2 == 0:
            ax.set_title(chosen_region + " kraj: " + "dialnica ("+ str(i//2+2018)+")")
        else:
            ax.set_title(chosen_region + " kraj: " + "dialnica ("+ str(i//2+2018)+")")
        ax.set_xlim([minx, maxx])
        ax.set_frame_on(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    d_2018.plot(markersize=1, ax=axs[0, 0], color="green")
    contextily.add_basemap(ax=axs[0, 0], crs="EPSG:3857", source=contextily.providers.Stamen.TonerLite)
    d_2019.plot(markersize=1, ax=axs[1, 0], color="green")
    contextily.add_basemap(ax=axs[1, 0], crs="EPSG:3857",  source=contextily.providers.Stamen.TonerLite)
    d_2020.plot(markersize=1, ax=axs[2,0], color="green")
    contextily.add_basemap(ax=axs[2,0], crs="EPSG:3857",  source=contextily.providers.Stamen.TonerLite)
    c1_2018.plot(markersize=1, ax=axs[0,1], color="red")
    contextily.add_basemap(ax=axs[0,1], crs="EPSG:3857", source=contextily.providers.Stamen.TonerLite)
    c1_2019.plot(markersize=1, ax=axs[1,1], color="red")
    contextily.add_basemap(ax=axs[1,1], crs="EPSG:3857", source=contextily.providers.Stamen.TonerLite)
    c1_2020.plot(markersize=1, ax=axs[2,1], color="red")
    contextily.add_basemap(ax=axs[2,1], crs="EPSG:3857",  source=contextily.providers.Stamen.TonerLite)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    if fig_location is not None and fig_location != "":
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru 
    
    gdf            : GeoDataFrame so zaznamami nehod, ktore chceme zobrazit
    fig_location   : ak nie je None a nie je prazdny string, tak ulozi obrazok na tuto cestu
    show_figure    : uzivatelska volba pre vykreslenie grafu
    """
    gdf = gdf.loc[(gdf["region"] == chosen_region) & (gdf["p36"] == 1)].to_crs("EPSG:3857")
    geometry = geopandas.GeoDataFrame(np.array([gdf["geometry"].x, gdf["geometry"].y]).transpose(), columns=["x", "y"])
    # pred tym sme vyskusali KMeans ale ten vykazoval mensiu schopnost lokalneho odhalenia nebezpecnych usekov
    # model = sklearn.cluster.KMeans(n_clusters=20) # 
    model = sklearn.cluster.AgglomerativeClustering(n_clusters=50) # cim viac zhlukov, tym mame moznost zachytit vysoko nehodove useky lokalnejsie
    predict = model.fit_predict(geometry)
    vc = pd.Series(predict).value_counts()
    color = [vc[val] for val in predict]
    _, axs = plt.subplots(1, 1, figsize=(8, 6))
    sc = plt.scatter(geometry["x"], geometry["y"],c=color, s=1)
    axs.set_title("Nehody v "+chosen_region + " kraji na cestach 1. triedy")
    axs.set_frame_on(False)
    axs.get_xaxis().set_visible(False)
    axs.get_yaxis().set_visible(False)
    contextily.add_basemap(ax=axs, crs="EPSG:3857", source=contextily.providers.Stamen.TonerLite)    
    plt.colorbar(sc, location="bottom", fraction=0.0592, pad=0.005)
    if fig_location is not None and fig_location != "":
        plt.savefig(fig_location)
    if show_figure:
        plt.show()

if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

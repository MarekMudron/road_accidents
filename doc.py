import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# textova definicia typov cestnych komunikacii
komunikacie = {
    0: "dialnica",
    1: "cesta 1. triedy",
    2: "cesta 2. triedy",
    3: "cesta 3. triedy",
    4: "uzol",
    5: "sledovana\nkomunikacia\n(vybrane mesta)",
    6: "miestna komunikacia",
    7: "polne cesty",
    8: "parkoviska,\nodpocivadla"
}


# k cislu kazdej z najcastejsich pricin je uvedena definicia
priciny = {
    503: "nedodrzanie vzdialenosti\nza vozidlom",
    508: "vodic sa nevenoval\nriadeniu",
    201: "neprisposobenie\nrychlosti",
    516: "iny druh nespravneho\nsposobu jazdy",
    511: "nezvladnutie riadenia\nvozidla"
}


def load_dataset(path):
    '''
        Nacita dataset zo specifikovaneho adresara

        path :  cesta k adresaru, odkial sa ma nacitat dataset
    '''
    df = pd.read_pickle(path)
    # predtym ako vratime dataset ho potrebujeme upravit do podoby
    # s ktorou budeme pracovat
    df = filter_dataset(df)
    return df


def filter_dataset(df):
    '''
        DataFrame df upravi do formy, s ktorou neskor pracujeme.
        Cas nehody prekonvertujeme na datetime, vytvorime novy 
        stlpec 'rok_mesiac' a vyfiltrujeme si len hromadne nehody
        v rokoch 2018-2020.

        df :    DataFrame, ktory sa bude upravovat
    '''
    # casove udaje konvertujeme na datetime
    df["p2a"] = pd.to_datetime(df["p2a"])
    # vytvorime stlpec s udajom vo formate ROK-MESIAC
    df["rok_mesiac"] = df["p2a"].dt.to_period("M").astype(str)
    # vyfiltrujeme si len hromadne nehody v rokoch 2018-2020
    # za hromadnu nehodu povazujeme nehodu s 3 a viac vozidlami
    df = df.loc[(df["p34"] > 2) & (df["p2a"].dt.year.isin([2018, 2019, 2020]))] # pocet vozidiel >= 3
    return df


def plot_monthly(df, show_fig=False, save_location=None):
    '''
        Funkcia zobrazi mesacny pocet hromadnych dopravnych nehod
        od januara 2018 do decembra 2020.

        df              :   DataFrame s datami o dopravnych nehodach
        show_fig        :   nastavuje, ci sa ma zobrazit vytvoreny graf
        save_location   :   cesta k adresaru, kam sa ma ulozit vytvoreny graf
    '''
    # zoskupenie vsetkych hromadnych nehod podla mesiaca
    nehody_za_mesiac = (df.groupby(df["rok_mesiac"])).size().reset_index(name="pocet")
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(nehody_za_mesiac["rok_mesiac"], nehody_za_mesiac["pocet"], color="steelblue")
    xt = []
    for i in range(0, len(nehody_za_mesiac), 3):
        xt.append(i)
    plt.xticks(xt, rotation=60)
    plt.title("Pocty hromadnych nehod mesacne")
    plt.grid(True)
    if save_location:
        plt.savefig(save_location)
    if(show_fig):
        plt.show()


def plot_conseq(df, show_fig=False, save_location=None):
    '''
        Funkcia zobrazi mesacny priebeh jednotlivych typov zdravotnych nasledkov hromadnych nehod.

        df              :   DataFrame s datami o dopravnych nehodach
        show_fig        :   nastavuje, ci sa ma zobrazit vytvoreny graf
        save_location   :   cesta k adresaru, kam sa ma ulozit vytvoreny graf
    '''
    # Nasledky hromadnych dopravnych nehod (mesacne)
    grp = df.groupby(df["rok_mesiac"])
    pocty_umrti_mesacne = grp["p13a"].sum().reset_index(name="mrtvi")
    pocty_tz_mesacne = grp["p13b"].sum().reset_index(name="tazko_zraneni")
    pocty_lz_mesacne = grp["p13c"].sum().reset_index(name="lahko_zraneni")
    xt = []
    for i in range(0, len(pocty_tz_mesacne), 3):
        xt.append(i)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.plot(pocty_umrti_mesacne["rok_mesiac"], pocty_umrti_mesacne["mrtvi"], label="mrtvi", color="chocolate")
    ax.plot(pocty_tz_mesacne["rok_mesiac"], pocty_tz_mesacne["tazko_zraneni"], label="tazko zraneni", color="steelblue")
    ax.plot(pocty_lz_mesacne["rok_mesiac"], pocty_lz_mesacne["lahko_zraneni"], label="lahko zraneni", color="darkseagreen")
    plt.grid(True)
    _ = ax.set_xticks(xt,rotation=90)
    _ = ax.tick_params(axis="x", rotation=40)
    _ = ax.set_yticks(np.arange(0, 320, 20))
    _ = ax.legend()
    _ = ax.set_title("Nasledky hromadnych dopravnych nehod (mesacne)")
    if save_location:
        plt.savefig(save_location)
    if(show_fig):
        plt.show()

def plot_yearly(df, show_fig=False, save_location=None):
    '''
        Funkcia zobrazi pocty dopravnych nehod na jednotlivych typoch cestnych komunikacii pre kazdy rok.

        df              :   DataFrame s datami o dopravnych nehodach
        show_fig        :   nastavuje, ci sa ma zobrazit vytvoreny graf
        save_location   :   cesta k adresaru, kam sa ma ulozit vytvoreny graf
    '''
    kom_count_by_year = df.groupby([df['p2a'].dt.year, "p36"])
    kom_count_by_year = kom_count_by_year.size().unstack(fill_value=0).stack()
    kom_count_by_year = kom_count_by_year.rename(index=komunikacie)
    # hromadne nehody podla druhu pozemnej komunikacie
    plt.rc('axes', axisbelow=True)
    kom_count_by_year.unstack(level=0).plot(kind="barh", figsize=(12,8), width=0.8, colormap="Accent", grid=True)
    plt.legend()
    plt.ylabel("")
    plt.title("Pocty dopravnych nehod v podla typu komunikacie za roky 2018-2020")
    plt.subplots_adjust(left=0.2)
    if save_location:
        plt.savefig(save_location)
    if(show_fig):
        plt.show()

def print_injuries(df):
    '''
        Funkcia vypise tabulku predstavujucu pocet jednotlivych zdravotnych nasledkov za jednotlive roky.

        df              :   DataFrame s datami o dopravnych nehodach
    '''
    tab = []
    for rok in [2018, 2019, 2020]:
        dat = df.loc[(df["p2a"].dt.year == rok)]
        tab.append([dat["p13a"].sum(), dat["p13b"].sum(), dat["p13c"].sum()])
    tab = pd.DataFrame(tab, columns=["mrtvi", "tazko zraneni", "lahko zraneni"])
    tab.index = [2018, 2019, 2020]
    print()
    print("Nasledky na zdravi pri hromadnych nehodach", end="\n\n")
    print(tab, end="\n\n")
    print("-"*50)

def plot_causes(df, year=2020, show_fig=False, save_location=None):
    '''
        Funkcia zobrazi graf popisujuci pocet nehod s konkretnou pricinou.
        Zobrazi pomer jednotlivych najcastejsich pricin pre dany typ komunikacie.

        df              :   DataFrame s datami o dopravnych nehodach
        show_fig        :   nastavuje, ci sa ma zobrazit vytvoreny graf
        save_location   :   cesta k adresaru, kam sa ma ulozit vytvoreny graf
    '''
    hl_priciny = df["p12"].value_counts()[:5].index.to_list()
    # z datasetu vyberieme len polozky, ktore su sposobene jednou z hlavnych pricin
    df = df.loc[df["p12"].isin(hl_priciny)]
    kom_pric_count = df.groupby([df['p2a'].dt.year, "p36", "p12"]).size().unstack(fill_value=0)
    kom_pric_count_yr = kom_pric_count.stack(level=0)[year]
    kom_pric_count_yr = kom_pric_count_yr.rename(index=komunikacie)
    kom_pric_count_yr = kom_pric_count_yr.unstack(level=1)
    kom_pric_count_yr = kom_pric_count_yr.rename(columns=priciny)
    kom_pric_count_yr.index
    plt.rc('axes', axisbelow=True)
    kom_pric_count_yr.plot(kind="bar", stacked=True, colormap="Accent", rot=20, figsize=(12, 8), width=0.8, grid=True)
    plt.legend(loc="upper left")
    plt.xlabel("")
    plt.subplots_adjust(bottom=0.2)
    plt.yticks(np.arange(0, 810, 50))
    if save_location:
        plt.savefig(save_location)
    if(show_fig):
        plt.show()

def print_stats(df, year=2020):
    '''
        Funkcia vypise vybrane statistiky pre uvedey rok

        df      : DataFrame s datami o dopravnych nehodach
        year    : rok, ktoreho statistiky, chceme zobrazit  
    '''
    print()
    print("STATISTIKY HROMADNYCH NEHOD PRE ROK", year, end="\n\n")
    df_yr = df.loc[df["p2a"].dt.year == year]
    ph = len(df_yr.index)
    print("Pocet hromadnych dopravnych nehod: ", ph)
    print("V priemere je to", ph//365, "hromadnych nehod za den.", end="\n\n")
    # alkohol v krvi > 1%
    drunk = df_yr["p57"].value_counts()[5]
    print("Pocet hromadnych dopravnych nehod, kde bol vodicovi zisteny alkohol v krvi > 1‰:", drunk)
    # percentualne vyjadrenie umrtnosti podla poctu nehod
    percent_umrtnost = round(df_yr["p13a"].sum()/len(df_yr.index)*100, 2)
    print("Umrtnost pri hromadnych dopravnych nehodach bola", percent_umrtnost, "%.", end="\n\n")
    # region s najvacsim poctom hromadnych nehod
    max_reg = df_yr["region"].value_counts().index[0]
    max_count = df_yr["region"].value_counts()[0]
    print("V pocte hromadnych dopravnych nehod dominoval region", max_reg, "s poctom", max_count, end=".\n\n")
    # zistenie kolko tvoritli hromadne dopravne nehody
    percent_hromadne = round(len(df_yr.index)/len(df.index)*100, 2)
    print("-"*50)

if __name__ == "__main__":
    df = load_dataset("accidents.pkl.gz")
    # dataset upravime podla nasich poziadaviek
    plot_monthly(df, save_location="fig_monthly.png")
    plot_conseq(df, save_location="fig_conseq.png")
    plot_yearly(df, save_location="fig_yearly.png")
    print_injuries(df)
    plot_causes(df, save_location="fig_causes.png")
    print_stats(df, 2020)
    
    

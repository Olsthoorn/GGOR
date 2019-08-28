##""********************************************************************************************************************
##  GXGCalc.py
##
##  VERSION: ArcGIS 9.0
##
##  AUTEUR: Co Drost GISwerk
##
##  REQUIRED ARGUMENTS: veel
##
##  OPTIONAL ARGUMENTS: beginjaar - startjaar van berekening
##                      eindjaar  - laatste jaar van berekening
##
##  DOEL:
##     Berekenen van GXG's van een records uit een featurelayer gegegeven een aantal argumenten
##
##  HISTORIE:
##     12 augutus 2008 Co Drost eerste versie op basis van VBA script inclusief flexibele peilen
##     4 september 2008 Co Drost aanpassingen na testen, export functionaliteit en logging
##               gewijzigde functionaliteit door kwelberekening op basis van WVP, C en gemiddeldpeil
##               export naar logging-file voor visualisatie in excel (featureclass shp/txt)
##     22 september 2008 Co Drost.
##            bugfixes: bepaling bereken periodes voor incomplete jaren
##                      kwel eenheden naar mm/dag voor input in formules
##                      kwelberekening voor flexibele peilen (bepaling gemiddeld peil voor incomplete jaren en
##                              correctie maaiveldhoogte)
##                      GVG berekening: deling door het juist aantal jaren
##            nieuwe functionaliteit:
##                      bepaling GVG voor hydrologische jaren in plaats van kalenderjaren
##            aanvullende documentatie en debugging
##
##
##*********************************************************************************************************************"""

##def DagAsDict(datum,neerslag,verdamping,netto_m,extreem,peil):
## return ['datum':datum,'neers

import sys,os,math,time, locale

# defining geoprocessor
try:
    import arcgisscripting
    pGP = arcgisscripting.create()
except:
    # Create the Geoprocessor object
    import win32com.client
    pGP = win32com.client.Dispatch("esriGeoprocessing.GpDispatch.1")


DEBUG = True

def debug(module,message):
    if DEBUG:
        strMessage = ">>> [" + module + "] " + message
        try:
            print (strMessage)
            pGP.addmessage(strMessage)
        except IOError:
            pass

def __D2L__(dd=0,mm=0,yy=0,sDate=''):
    # transform date to long
    if len(sDate) > 0:
        lDate = sDate.split('-')
        lng = int(lDate[2]) * 10000 + int(lDate[1]) * 100 + int(lDate[0])
    else:
        lng = int(yy) * 10000 + int(mm) * 100 + int(dd)
    return lng

def __Dag__(lng):
    return lng % 100

def __IsZomer__(lng):
    maand = int(str(lng)[4:6])
    if maand > 3 and maand < 10:
        return True
    else:
        return False

class GxGCalc:

    def ReadNeerslag(self,neerslagfile):
        """
        Inlezen van neerslag/verdampingsreeksen uit bestand. Waarden met decimale punt.
        Datums in format dd-mm-jjjj
        Format: datum  neerslag  verdamping (gescheiden door spaties of tab)
        """
        # read rainfall data from file
        sMod = 'GxGCalc::ReadNeerslag'
        # start with empty reeks
        reeks = []
        # check if file exists and read contents
        if not os.path.exists(neerslagfile):
            debug(sMod,'File not found: ' + neerslagfile)
            return -1
        fi = open(neerslagfile,'r')
        lines = fi.readlines()
        fi.close()
        # place content in dictionary
        i = 0
        debug(sMod,'Number of lines: ' + str(len(lines)))
        self.minjaar = 0
        self.maxjaar = 0
        for dag in lines:
            i += 1
            sdag = dag.strip()
            if sdag[0] != ';':
                lstdag = dag.split()
                if (len(lstdag) == 3):
                    dd = __D2L__(sDate=lstdag[0])
                    jaar = int(dd / 10000)
                    if self.minjaar == 0 and self.maxjaar == 0:
                        self.minjaar = jaar
                        self.maxjaar = jaar
                    else:
                        self.minjaar = min(self.minjaar,jaar)
                        self.maxjaar = max(self.maxjaar,jaar)
                    neerslag = float(lstdag[1])
                    verdamping = float(lstdag[2])
                    dict = {'datum':dd,'neerslag':neerslag,'verdamping':verdamping,'netto_m':(neerslag-verdamping)/1000,'peil':-9999}
                    reeks.append(dict)
                else:
                    debug(sMod,'fout op regel ' + str(i))
        debug("ReadNeerslag","Beginjaar/EindJaar:  " + str((self.minjaar,self.maxjaar)))
        self.reeks = reeks
        return 0

    ##-------------------------------------------------------------------------------------------------
    def ZetPeilen(self,ZP,WP,MV,FlexFile):
    ##-------------------------------------------------------------------------------------------------
        sMod ='GxGCalc::ZetPeilen'
        # controle of er sprake is van flexibel peil
        bIsFlex = FlexFile != ''
        peilen = {}
        if bIsFlex:
            if not os.path.exists(FlexFile):
                debug(sMod,'File ' + FlexFile+ ' not found')
                return -1
            else:
                fi = open(FlexFile,'r')
                lines = fi.readlines()
                debug(sMod,'Reading ' + str(len(lines)) + ' lines...')
                beginjaar = None
                eindjaar = None
                for ln in lines:
                    sln = ln.strip()
                    if sln[0] != ';':
                        lln = sln.split()
                        # print sln,len(lln)
                        if len(lln) == 2:
                            dd = __D2L__(sDate=lln[0])
                            if beginjaar == None:
                                beginjaar = dd / 10000
                            eindjaar = dd / 10000
                            peilen[dd] = float(lln[1])
                debug("ZetPeilen","Laatste dag in flexpeilen: " + str(dd))
                self.beginjaar = beginjaar
                self.eindjaar = eindjaar

        for el in self.reeks:
            peil = 0
            if bIsFlex:
                if peilen.has_key(el['datum']):
                    peil = peilen[el['datum']]
                else:
                    peil = -9999
            else:
                if __IsZomer__(el['datum']):
                    peil = ZP
                else:
                    peil = WP
            el['peil'] = peil - MV
            el['extreem'] = -9999
            ##print el['peil']

    ##-------------------------------------------------------------------------------------------------
    def print_reeks(self):
        for i in self.reeks:
            debug('GxGCalc::print_reeks',str(i))

    ##-------------------------------------------------------------------------------------------------
    def BerekenKwel(self,C,Wvp):
#    """
#    Berekenen van kwel op basis van een gemiddeld peil bij flexibele peilen
#    """

        # eerst berekenen gemiddeld peil over de hele periode
        peilen = []
        for i in self.reeks:
            peil = i['peil']
            if peil > -30:
                peilen.append(peil)
        if len(peilen) == 0:
            return -9999
        peilgem = sum(peilen)/len(peilen)
        debug("BerekenKwel","Gemiddeld peil: " + str(peilgem))
        # berekenen kwel op basis van verschil met stijghoogte in het 1ste WVP gedeeld door de weerstand van de deklaag
        kwel = (Wvp-peilgem) / C
        return kwel

    ##-------------------------------------------------------------------------------------------------
    def ReeksClip(self,beginjaar='#',eindjaar='#'):
        if beginjaar == '#' and eindjaar == '#':
            self.start = 0
            self.eind = len(self.reeks) - 1
            self.beginjaar = self.reeks[0]['datum'] / 10000
            self.eindjaar = self.reeks[-1]['datum'] / 10000
            return self.reeks
        if beginjaar == '#':
            beginjaar = self.reeks[0]['datum']  / 10000
        if eindjaar == '#':
            n = len(self.reeks)-1
            eindjaar = self.reeks[n]['datum']  / 10000
        start = 0
        eind = len(self.reeks)
        n = 0
        for i in self.reeks:
            n += 1
            jaar =  i['datum'] / 10000
            if jaar < beginjaar or i['peil'] < -100:
                if start == n - 1:
                    start = n
            if jaar <= eindjaar and i['peil'] > -100:
                eind = n
        self.start = start
        self.eind = eind -1
        self.beginjaar = self.reeks[start]['datum'] / 10000
        self.eindjaar = self.reeks[eind-1]['datum'] / 10000
        debug("ReeksClip","Begin/eind: " + str((start,eind-1)))
        debug("ReeksClip","Begin/eind datum: " + str((self.reeks[start]['datum'],self.reeks[eind-1]['datum'])))
        return # self.reeks[start-1:eind]

    ##-------------------------------------------------------------------------------------------------
    def Bereken(self,c,L,fi0,K,ZP,WP,FP,MV,Mu,Wvp,beginjaar='#',eindjaar='#'):
    ##-------------------------------------------------------------------------------------------------
        rest = 0.001  # afkappen berekenen indien verschil nog maar 1 mm is
        cH = 6 # dikte van de deklaag
        cRefniv = -10
        cFractieL = 0.33
        c = float(c)

        starttime = time.time()

        # eerste vaste berekeningen
        beta = math.sqrt(1 / (K * cH * c))
        c1 = (math.cosh(beta * L) - 1) / (beta * math.sinh(beta * L))
        f = 1 / ((-2 * c * c1 * Mu / L) + Mu * c + 1)
        signaal = 0.0

        kwel_ber = 0
        iteratie = 0
        gereed = False
        stap = 0.5
        rico = 1

        # Zetten begin en eindpunt iteratie op default waarden
        r = self.ReeksClip()
        if beginjaar == '#': beginjaar = self.beginjaar
        if eindjaar == '#': eindjaar = self.eindjaar
        beginjaar=int(beginjaar)
        eindjaar=int(eindjaar)

        #peilen zetten op basis van datum op winter, zomer of flexpeil uit bestand
        self.ZetPeilen(ZP,WP,MV,FP)

        # zetten van begin en eind op parameter
        self.beginjaar  = max(self.beginjaar,beginjaar)
        self.eindjaar  = min(self.eindjaar,eindjaar)
        ##print 'XXX',self.beginjaar,self.eindjaar
        r = self.ReeksClip(self.beginjaar,self.eindjaar)
        ##print 'E',self.beginjaar,self.eindjaar

        #debug('Berekenen','C: ' + str(c))
        if (c == 0):
            raise [Exception,"C = 0"]

        # pGP.addmessage(str(self.start) + ' ' + str(self.eind))

        Kwel = 0.0
        # Berekenen kwel op basis van stijghoogte en weerstand van de deklaag
        if FP == '':
            Kwel =  (Wvp - ((ZP + WP) / 2)) / c
        else:
            # Bereken kwel op basis van verschil in gemiddelde peil en
            # hoogte WVP/ c (weerstand van de deklaag).
            # omdat het gemiddelde peil gecorrigeerd is ten opzichte
            # van MV ook WVP corrigeren t.o.v. MV
            Kwel = self.BerekenKwel(c,Wvp-MV)
        # kwel omzetten naar mm per dag
        Kwel = Kwel * 1000
        # tonen tussen waarden
        debug('Berekenen','Berekende Kwel = ' + str(Kwel))
        debug('Berekenen','controle: '+ str(self.start) + ' -  ' +  str(self.eind))

        while iteratie < 200 and gereed != True:
            iteratie = iteratie + 1
            i = 0
            for pDag in self.reeks[self.start:self.eind+1]:
                i = i + 1
                #berekenen verdamping penman....niet belangrijk
                ##pDag.verdamp_penman = pDag.verdamp_makking * Penman(CInt(maand))

                peil = pDag['peil']
                #Als peil niet goed is dan stoppen
                if (peil == -9999):
                    self.GLG = peil
                    self.GHG = peil
                    self.GLG = peil
                    return -1

                hh = math.sinh(beta * L)
                hh2 = math.sinh(0.5 * beta * L)
                nn = (K * cH * -fi0 * hh - 2 * K * cH * (peil - fi0) * hh2) * (beta**2)  / (hh - 2 * hh2)
                ##nn = (K * H * -fi0 * Sinh(beta * L) - 2 * K * H * (peil - fi0) * Sinh(0.5 * beta * L)) * beta ^ 2 / (Sinh(beta * L) - 2 * Sinh(0.5 * beta * L))
                signaal = min(signaal + f * (pDag['netto_m'] - signaal), nn)
                ##=+P39/($E$21^2)+$E$13*$E$14*$E$16
                fi0a = signaal / (beta ** 2) + K * cH * fi0
                ###=+((($E$13*$E$14*$F56-$Q56)*SINH($E$21*$E$12*(1-H$12))+($E$13*$E$14*$F56-$Q56)*SINH($E$21*$E$12*H$12))/SINH($E$21*$E$12)+$Q56)/($E$13*$E$14)
                extreem = (((K * cH * peil - fi0a) * math.sinh(beta * L * (1 - cFractieL)) + (K * cH * peil - fi0a) * math.sinh(beta * L * cFractieL)) /
                           math.sinh(beta * L) + fi0a) / (K * cH)
                ###=ALS(R39>$E$20;($E$20-$E$16)/$E$11*$E$8;($E$21*(COSH($E$21*$E$12)-1)/SINH($E$21*$E$12)*($E$13*$E$14*($F39+$F39)-2*Q39)+P39*$E$12)*$E$8/$E$12)
                if (extreem > 0):
                    wegzijging = -fi0 / (c / 10000)
                else:
                    ###(beta*(COSH(beta*L)-1)/SINH(beta*L)*(k*H*(peil+peil)-2*)+P39*$E$12)*$E$8/$E$12)
                    wegzijging = (beta * (math.cosh(beta * L) - 1) / math.sinh(beta * L) * (K * cH * (2 * peil) - 2 * fi0a) + signaal * L) * 10000 / L

                pDag['extreem'] = extreem
                kwel_ber = kwel_ber + (-wegzijging / 10)

            kwel_ber = kwel_ber / i

            # controleren of klaar anders verkleinen van de stap
            if abs(kwel_ber - Kwel) < rest:
              gereed = True
            else:
                if kwel_ber > Kwel:
                    if (rico > 0):
                        rico = -1
                        stap = stap / 2
                else:
                    if (rico < 0):
                      rico = 1
                      stap = stap / 2
                fi0 = fi0 + (rico * stap)
            #debug("iteratie",str(fi0) + ' ' + str(kwel_ber) + ' ' + str(stap) + ' ' + str(rico))

        #loop

        #als de iteratie gelukt is dan Gxg bepalen
        self.GLG = 0
        self.GHG = 0
        self.GVG = 0
        if gereed:
            # GxGBerekenen
            debug('GxG berekenen...',str(time.time() - starttime) + ' seconden)')

      #Debug.Print Format(Now, "hh:mm:ss") & "  " & TT - Timer

    ##-------------------------------------------------------------------------------------------------
    def GxG(self):
    ##-------------------------------------------------------------------------------------------------
        """
        Berekenen van GXG over de gegeven periode
        GLG : periode gemiddelde van de laagste 3 waardes per jaar van gws op 14 of 28ste van de maand
        GLG : periode gemiddelde van de hoogste 3 waardes per jaar van de gws op 14 of 28ste van de maand
        GVG : gemiddelde van de GVG's op 1 april van de gws
        """

        GLG,GVG,GHG = 0,0,0

        GVGsum = 0.0
        GHGsum = []
        GLGsum = []
        GVGaantal = 0

        # eerste jaar overslaan vanwege aanloop
        for jaar in range(self.beginjaar+1,self.eindjaar+1):
        ##for jaar in range(2001,2007):
            ## print jaar
            GXGs = []
            aantal = 1
            for el  in self.reeks:
                if el['extreem'] > -100:
                    jr = int(el['datum'] / 10000)
                    maand = el['datum'] / 100 % 100
                    dag = el['datum'] % 100
                    if jr == jaar:
                        if dag == 14  or dag == 28:
                            GXGs.append(el['extreem'])
                        elif maand == 4 and dag == 1:
                            GVGsum += el['extreem']
                            GVGaantal += 1
            GXGs.sort()
            #debug('GxG',str(jaar) + '\t %f \t %f \t %f' % tuple(GXGs[:3]))
            GLGsum.extend(GXGs[:3])
            GHGsum.extend(GXGs[-3:])
        # 1 jaar minder vanwege vervallen eerste jaar
        GLG = sum(GLGsum) / len(GLGsum)
        GHG = sum((GHGsum)) / len(GHGsum)
        GVG = GVGsum / GVGaantal
        return {'GLG':GLG,'GVG':GVG,'GHG':GHG}

    ##-------------------------------------------------------------------------------------------------
    def GxG_Hydr(self):
    ##-------------------------------------------------------------------------------------------------
        """
        Berekenen van GXG over de gegeven periode overde hydrologische jaren
        GLG : periode gemiddelde van de laagste 3 waardes per jaar van gws op 14 of 28ste van de maand
        GLG : periode gemiddelde van de hoogste 3 waardes per jaar van de gws op 14 of 28ste van de maand
        GVG : gemiddelde van de GVG's op 1 april van de gws
        """

        GLG,GVG,GHG = 0,0,0

        GVGsum = 0.0
        GHGsum = []
        GLGsum = []
        GVGaantal = 0

        # eerste jaar overslaan vanwege aanloop
        # laatste jaar niet meetellen omdat dit maar tot 1 april loopt.
        for jaar in range(self.beginjaar+1,self.eindjaar):
        ##for jaar in range(2001,2007):
            ## print jaar
            GXGs = []
            aantal = 1
            startdatum = jaar * 10000 + 401
            einddatum = (jaar + 1) * 10000 + 331
            #debug("Gxg_Hydr",str((startdatum,einddatum)))
            for el  in self.reeks:
                if el['datum'] >= startdatum and el['datum'] <= einddatum and el['extreem'] > -100:
                    maand = el['datum'] / 100 % 100
                    dag = el['datum'] % 100
                    if dag == 14  or dag == 28:
                        GXGs.append(el['extreem'])
                    elif maand == 4 and dag == 1:
                        GVGsum += el['extreem']
                        GVGaantal += 1
            GXGs.sort()
            #debug('GXGtest',str(jaar) + ' : ' + str(len(GXGs)))
            #debug('GxG',str(jaar) + '\t %f \t %f \t %f' % tuple(GXGs[:3]))
            GLGsum.extend(GXGs[:3])
            GHGsum.extend(GXGs[-3:])
        # 1 jaar minder vanwege vervallen eerste jaar
        GLG = sum(GLGsum) / len(GLGsum)
        GHG = sum((GHGsum)) / len(GHGsum)
        GVG = GVGsum / GVGaantal
        return {'GLG':GLG,'GVG':GVG,'GHG':GHG}


def test():
    rks = GxGCalc()
    rks.ReadNeerslag('D:/projdata/waternet/GxG/neerslag81-07.txt')
    #rks.ZetPeilen(0,0,0,'D:/projdata/waternet/GxG/flexpeil/63-7.txt')
    rks.Bereken(2193, 96.92, 1, 0.2, -2.5,-2.4,'d:/projdata/waternet/gxg/flexpeil/63_7.txt',-2.3, 0.25, -4.35,'#','#')
    debug('test',str(rks.GxG()))
    debug('test',str(rks.GxG_Hydr()))


try:

    ## r = rks.ReeksClip(2000,2008)
    ##print r[1]['datum']
    ##print r[-1]['datum']

    #rks = GxGCalc()
    #rks.ReadNeerslag('D:/projdata/waternet/GxG/neerslag96-04.txt')
    #rks.ReadNeerslag('D:/projdata/waternet/GxG/neerslag81-07.txt')
    # rks.ZetPeilen(0,0,0,'D:/projdata/waternet/GxG/flexpeil/63_7.txt')
    #def Bereken(self,c,L,fi0,K,ZP,WP,FP,MV,Mu,Wvp,beginjaar=2000,eindjaar=2003):
    #rks.Bereken(2193, 96.92, 1, 0.2, -2.5,-2.4,'d:/projdata/waternet/gxg/flexpeil/63_7.txt',-2.3, 0.25, -4.35,'#','#')
    #rks.Bereken(2175,81.99, 1, 0.2, -2.5,-2.4,'d:/projdata/waternet/gxg/flexpeil/63_7.txt',-2.06, 0.25, -4.0842,1990,2000)
    #rks.Bereken(2193, 96.92, 1, 0.2, -2.5,-2.4,'',-2.3, 0.25, -4.35,2000,2005)
    #rks.Bereken(1996.083008, 74.77, 1, 0.2, -2.94,-2.99,'',-2.06,0.25,-3.871027,2000,2003)
    #print rks.GxG()
    #print rks.GxG_Hydr()

    if len(sys.argv) < 2:
        raise [Exception,"Geen argumenten"]

    flyr = sys.argv[1]
    fldZP = sys.argv[2]
    fldWP = sys.argv[3]
    fldFP = sys.argv[4]
    fldMV = sys.argv[5]
    fldL = sys.argv[6]
    fldC = sys.argv[7]
    fldWvp = sys.argv[8]
    fldK = sys.argv[9]
    fldMu = sys.argv[10]
    fldGLG = sys.argv[11]
    fldGHG = sys.argv[12]
    fldGVG = sys.argv[13]
    fnNeerslag = sys.argv[14]
    fnFPDir = sys.argv[15]
    nBeginjaar = sys.argv[16]
    nEindjaar = sys.argv[17]

    rks = GxGCalc()
    rks.ReadNeerslag(fnNeerslag)

    #debug('FP',fldFP)
    #debug('ZP',fldZP)
    #debug('WP',fldWP)
    #debug('MV',fldMV)
    #debug('Wvp',fldWvp)
    #debug('C',fldC)
    #debug('L',fldL)
    #debug('K',fldK)
    #debug('Mu',fldMu)
    #debug('Neerslag',fnNeerslag)

    desc = pGP.describe(flyr)
    catalogpath = desc.CatalogPath
    # fldOID = desc.FeatureClass.OIDFieldName
    ## pGP.addmessage('OIDfield: ' + fldOID)
    descshp = pGP.describe(catalogpath).DataType.lower()
    if descshp == 'shapefile':
        logfile = catalogpath[:-3] + 'txt'
        logdict = None
        header = None
    else:
        logfile = None

    cur = pGP.updatecursor(flyr)


    aantal = pGP.GetCount_Management(flyr)
    pGP.addmessage('Processing ' + str(aantal) + ' records...')

    aantal = 0
    row = cur.next()
    while row:
        aantal += 1
        # reading field values for calculation
        ##oid = row.getvalue(fldOID)
        oid = aantal
        debug("Processing...","record " + str(oid))
        ZP = row.getvalue(fldZP)
        WP = row.getvalue(fldWP)
        FP = row.getvalue(fldFP).strip()
        MV = row.getvalue(fldMV)
        Wvp = row.getvalue(fldWvp)
        L = row.getvalue(fldL)
        K = row.getvalue(fldK)
        C = row.getvalue(fldC)
        Mu = row.getvalue(fldMu)
        # checking for flexibel peil
        fnFP = ''
        if FP != '':
            fnFP = fnFPDir+ '/' + FP + '.txt'
            if not os.path.exists(fnFP):
                pGP.AddWarning('Flexibelpeil [' +FP+ '] niet gevonden als ' +
                               fnFP)
                raise [Exception, 'FP not found']
        # doing calculatations
        ## def Bereken(self,c,L,fi0,K,ZP,WP,FP,MV,Mu,Kwel):
        # controleren input!
        GXGs = {'GHG':-9999,'GLG':-9999,'GVG':-9999}
        if C <= 0:
            pGP.addwarning('Record ' + str(oid) + ':  C <= 0')
        else:
            rks.Bereken(C, L, 1.0, K, ZP, WP,fnFP,MV, Mu, Wvp,nBeginjaar,nEindjaar)
            # berekenen GxG uit fi0
            ## op basis van normale jaren
            ## GXGs = rks.GxG()
            ## op basis van hydrologische jaren
            GXGs = rks.GxG_Hydr()
            # do logging
            if logfile and aantal < 10:
                if not logdict:
                    logdict = []
                    header = 'datum'
                    for i in range(len(rks.reeks)):
                        logdict.append([rks.reeks[i]['datum']])
                # toevoegen reeks aan logging
                header += '\t' + str(oid)
                for i in range(len(rks.reeks)):
                    # debug('logfile',str(rks.reeks[i]['datum']) + ' ' + str(rks.reeks[i]['datum']))
                    if rks.reeks[i].has_key('extreem'):
                        logdict[i].append(rks.reeks[i]['extreem'])
                    else:
                        logdict[i].append(0)

            # setting calculculated values
            ##pGP.addmessage(fldGHG+' ' + str(GXGs['GHG']))
        row.setvalue(fldGHG,GXGs['GHG'])
        row.setvalue(fldGLG,GXGs['GLG'])
        row.setvalue(fldGVG,GXGs['GVG'])
        cur.updaterow(row)
        # next record in layer
        row = cur.next()

    # als logging dan wegschrijven naar file
    if logfile:
        if logdict:
            tf = open(logfile,'w')
            tf.write(header+'\n')
            for r in logdict:
                tf.write(str(r[0]))
                for h in range(1,len(r)):
                    tf.write('\t' + locale.format('%2.2f',r[h]))
                tf.write('\n')
            tf.close()
            pGP.addwarning('Logging weggeschreven in ' + logfile)


    # remove update cursor for unlocking featurelayer
    del cur


except [Exception, ErrorDesc]:
    # handle the errors here. if the point creation fails, want to keep track of which point failed (easier to fix the
    # text file if we do
    pGP.AddError(str(ErrorDesc))
    debug('gw_pro_pnt.py:',str(ErrorDesc))
    pGP.AddError(pGP.getmessages(2))
    # removing eventualy locked items
    try:
        del cur
    except:
        print ('')

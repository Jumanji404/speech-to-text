# project-chatbot-danonino-eating-chupacabra

Un pacient ne povestește:

Am fost azi dimineață la un control medical și doctorul, după o serie de investigații, a început să-mi spună ce ar
trebui să fac pentru a mă simți mai bine, ce medicamente să iau, în ce ordine, etc. Dar mi-a zis așa multe deodată încât
s-au amestecat toate și e cam haos în mintea mea. Mi-a dat o rețetă, dar e cu scris de doctor…

Un medic rezident ne povestește:

Au trecut deja două săptămâni din rezidențiatul meu. Am avut șansa să intru într-o echipă excelentă de medici
profesioniști de la care pot învăța foarte multe. Dar îmi transmit așa multe informații, încât mi-e greu să le rețin și
să le asimilez. Nu apuc să notez mai nimic…

# Problemă:

Cum pot înțelege limbajul vocal al medicilor? Să îl recunosc și să îmi iau notițe.


---

# Preprocesarea Datelor:

## Prelucrarea datelor audio

Cu toate ca inteligenta artificiala evolueaza pe zi ce trece, inca nu s-a ajuns la intelegerea si procesarea performanta
a fisierelor audio in forma lor naturala. Astfel, pentru a procesa fisiere audio am decis sa transformam datele intr-o
forma bine stapanita de algoritmii de AI, anume imagini .
> Spectrograma = Imaginea obtinuta prin prelucrarea fisierului audio urmarind frecventele si intensitatile acestuia.

Pentru a transforma fisiere audio in spectrograme am folosit tensorflow_io si modulul de preprocessing din keras.
Procesul pe care l-am urmat este:

1. Generarea de waveform-uri dupa frecventa fisierelor audio:
     ````
    audio = tfio.audio.decode_mp3(audioBinary)
    return tf.squeeze(audio, axis=-1)
   ````
2. Generarea spectrogramelor pe baza waveform-urilor:
    ````
    waveform = tf.cast(waveform, tf.float32)
    equalLength = tf.concat([waveform, zeroPadding], 0)
    spectrogram = tf.signal.stft(
    equalLength, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
   ````
3. Salvarea spectrogramelor cu numele fisierului audio pe baza caruia au fost create:
    ````
    fig, axes = plt.subplots(1, figsize=(12, 12))
    axes.set_title('Spectrogram for ' + spectrogramTitle)
    plotSpectrogram(spectrogram.numpy(), axes)
    fileName=str(spectrogramName.split(".")[0])+".jpg"
    plt.savefig(destinationFolder+"/"+fileName)
   ````
### Modul de reprezentare a unui waveform respectiv al  spectrogramei obtinute pe baza acestuia:

![](https://cdn.discordapp.com/attachments/722812184002035807/854344725296316437/unknown.png)
    
## Prelucrearea datelor text

Folosind o retea neuronala, datele text (outputul modelului) ar trebui transformate in valori numerice. 
Pentru aceasta am folosit tot libraria preprocessing din keras, mai exact Tokenizer-ul ,acesta fiind o functie de hashing pentru transformarea valorilor de tip string in valori numerice ("Ana" => 2) 
````
        # dataString= Lista de cuvinte a propozitiei
        tokenizer.fit_on_texts(dataString)
        sequences = tokenizer.texts_to_sequences(dataString)
        vocabularySize = len(tokenizer.word_index) + 1
        sequences=pad_sequences(sequences, maxlen=vocabularySize)
````
Pentru a introduce  date tokenizate in layerul de output al retelei neuronale,ar trebui ca acestea sa aiba o lungime identica.<br>
Astfel, a fost necesara gasirea unei codificari care sa indeplineasca aceasta nevoie.<br>
Am decis sa codificam propozitiile intr-un vector cu valori binare cu lungimea vocabSize(numarul de cuvinte distincte din setul de date de antrenament)<br>
Metoda de codificare consta in a pune valoarea 1 la indexul id-ului dat de Tokenizer pentru un anumit cuvant -1, pentru fiecare cuvant din propozitie, restul valorilor din vector fiind 0.

>Spre exemplu daca avem propozitia: "Ana are mere" cu tokenizarea: [2,4,6] iar vocabSize=7,
> vectorul binar asociat acestei tokenizari ar fi [0,1,0,1,0,1,0]
````
        rez = np.array([np.array([0 for _ in range(vocabularySize)]) for _ in range(len(dataString))])
        for i in range(len(sequences)):
        for j in range(len(sequences[i])):
        if sequences[i][j]!=rez[i][j]:
        rez[i][sequences[i][j]-1]=1
        return rez,vocabularySize
````
# AutoEncoder
Avand in vedere dimensiunea foarte mare a valorii vocabSize-ului pentru datele de antrenament (aprox. 6500) iar in medie numarul de cuvine/propozitie este 7,
din punct de vedere al unei clasificari binare, numarul valorilor egale cu 1 este mult mai mic decat numarul valorilor egale cu 0, acest lucru facand foarte dificil
ca modelul nu doar sa ghiceasca corect valorile de 1 ci efectiv sa aiba outputul 1 (Daca prezice constant doar 0 modelul  ar avea un loss foarte mic (aprox. 10^(-2)) acest lucru ducand la o acuratete de aprox. 98%).
<br>
>Solutie: Implementarea operatiei de Dimensionality Reduction (cu scopul de a rezolva problema de sparsity) folosind o arhitectura de tip AutoEncoder
<br>
 
>Acuratetea AutoEncoderului: <br>
> Test accuracy for AutoEncoder (sentences): 0.9355692850838482<br>
Test accuracy for AutoEncoder (ones): 0.8435754189944135<br>
Test accuracy for AutoEncoder (zeros): 0.9996523443165235<br>

![](https://media.discordapp.net/attachments/722812184002035807/854353563910930483/autoencoder.jpg?width=104&height=473)

# ResNet
Avand in vedere faptul ca diferenta dintre 2 spectrograme ale 2 propozitii este mult mai mica decat diferenta semantica 
a celor 2 propozitii (d.p.d.v. al unei probleme de Computer Vision patternurile pe care trebuie sa le identifice modelul
in imagini sunt mult mai dificile decat antrenarea o problema de clasificare a semnelor de circulatie).<br>
Prin urmare solutia a fost de a implementa o arhitectura Deep de tipul ResNet (aceasta fiind o solutie a fenomenului de Vanishing Gradient Descent
care este des intalnit in contextul retelelor neuronale Deep).
<br>

![](https://media.discordapp.net/attachments/722812184002035807/854359257101238302/unknown.png?width=189&height=472)

#Rezulatele modelului

>Real output: ['Trebuie', 'să', 'ne', 'bazăm', 'pe', 'acest', 'lucru.']  <br> Predicted Output: ['să', 'acest', 'trebuie', 'pe', 'ne', 'lucru.', 'bazăm']<br>
> <br>
> Real output: ['Prin', 'urmare,', 'am', 'votat', 'în', 'favoarea', 'acesteia.']   <br> Predicted Output: ['în', 'am', 'prin', 'urmare,', 'votat', 'favoarea', 'acesteia.']
> <br> <br>
> Real output: ['Aceştia', 'sunt', 'absolut', 'copleşiţi', 'de', 'jungla', 'etichetelor.']  <br>  Predicted Output: ['de', 'sunt', 'absolut', 'aceştia', 'copleşiţi', 'jungla', 'etichetelor.']
  <br> <br> 
> Real output: ['Noi', 'nu', 'vom', 'vota', 'în', 'favoarea', 'ei.']   <br>  Predicted Output: ['nu', 'în', 'vom', 'favoarea', 'noi', 'vota', 'ei.']
> <br> <br>
> Real output: ['Vă', 'mulţumesc', 'pentru', 'ultimii', 'cinci', 'ani', 'de', 'colaborare.']  <br>  Predicted Output: ['de', 'pentru', 'vă', 'mulţumesc', 'ani', 'cinci', 'ultimii', 'colaborare.']
<br> <br>
> Real output: ['Permiteți-mi', 'însă', 'să', 'introduc', 'și', 'un', 'aspect', 'umanitar.']   <br> Predicted Output: ['să', 'mai', 'multe', 'doresc', 'adresez', 'comisar', 'întrebări.', 'doamnei']
<br> <br>
> Real output: ['Există', 'vreo', 'explicaţie', 'pentru', 'această', 'absenţă?']  <br>  Predicted Output: ['pentru', 'această', 'există', 'vreo', 'explicaţie', 'absenţă?']
<br> <br>
> Real output: ['Croația', 'nu', 'a', 'finalizat', 'încă', 'aceste', 'reforme.']   <br>  Predicted Output: ['la', 'se', 'referă', 'meu', 'ultimul', 'comentariu', 'exporturi.']
<br> <br>
> Real output: ['Medicamentele', 'falsificate', 'sunt', 'ucigași', 'din', 'umbră.']  <br>  Predicted Output: ['sunt', 'din', 'medicamentele', 'falsificate', 'ucigași', 'umbră.']
<br> <br>
> Real output: ['Finanţarea', 'sistemului', 'medical', 'nu', 'este', 'de', 'neglijat.']  <br> Predicted Output: ['de', 'este', 'nu', 'sistemului', 'finanţarea', 'medical', 'neglijat.']
<br> <br>

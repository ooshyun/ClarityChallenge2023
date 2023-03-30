```
clarity_data/train
    interferers
        music
        noise
        speech
    rooms
        ac
        HOA_IRs
        rpf
    scenes
    targets

metadata/
    hrir_data.json
        selected_channels
        matrix
        weights

    listeners.json			
        name
        audiogram_cfs
        audiogram_levels_l
        audiogram_levels_r

    masker_music_list.json
        track
        artist
        album
        file
        duration
        dataset
        fs
        type
        license
        nsamples

    masker_nonspeech_list.json	
        ID
        class(noise type)
        source
        file
        dataset
        type
        nsamples
        duration
        fs

    rooms.dev.json	
        name
        dimenstions
        target
            positions
            view_vector
        lister
            positions
            view_vector
        interferers
            positions
            view_vector

    scenes.train.json, scenes.dev.json
        dataset
        room
        scence
        target
            name
            time_start
            time_end
        duration
        interferers(1, 2, 3 position, selective ex. 1, 3 /1, 2, / 1, 2, 3)
            position
            time_start
            time_end
            type
            name
            offset
        SNR
        listener
            rotation
                sample
                angle
            hrir_filename

    masker_speech_list.json
        utternaces
        dataset
        type
        speaker
        nsampels
        duration
        fs

    rooms.train.json	
        name
        dimensions
        target
            position
            view_vector
        listener
            position
            view_vector
        interferers(1, 2, 3)
            position

    scenes_listeners.dev.json		
        scene name(S06005)
        hearing loss type(L0023, num=3)
                
    target_speech_list.json
        prompt
        prompt_id
        speaker
        wavfile
        index
        dot
        sex
        nsamples
        duration
        fs

    mtg_jamendo_music_licenses.txt	
```
// Songwriting Assistant Web Interface

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';
const POLLING_INTERVAL = 1000; // 1 second

// DOM Elements
document.addEventListener('DOMContentLoaded', () => {
    // Lyrics generation
    const lyricsForm = document.getElementById('lyrics-form');
    const artistInput = document.getElementById('artist');
    const titleInput = document.getElementById('title');
    const themeInput = document.getElementById('theme');
    const lyricsPromptInput = document.getElementById('lyrics-prompt');
    const temperatureInput = document.getElementById('temperature');
    
    // Chord progression
    const chordForm = document.getElementById('chord-form');
    const keyInput = document.getElementById('key');
    const sectionInput = document.getElementById('section');
    const chordResult = document.getElementById('chord-result');
    
    // Workspace
    const workspaceTitle = document.getElementById('workspace-title');
    const workspaceLyrics = document.getElementById('workspace-lyrics');
    const saveButton = document.getElementById('save-song');
    
    // Tools
    const rhymeForm = document.getElementById('rhyme-form');
    const rhymeWordInput = document.getElementById('rhyme-word');
    const rhymeResults = document.getElementById('rhyme-results');
    
    const imageryForm = document.getElementById('imagery-form');
    const imageryThemeInput = document.getElementById('imagery-theme');
    const imageryResults = document.getElementById('imagery-results');
    
    const searchForm = document.getElementById('search-form');
    const searchArtistInput = document.getElementById('search-artist');
    const searchTitleInput = document.getElementById('search-title');
    const searchResults = document.getElementById('search-results');
    
    // Saved songs
    const savedSongsList = document.getElementById('saved-songs-list');
    
    // Event listeners
    lyricsForm.addEventListener('submit', handleLyricsGeneration);
    chordForm.addEventListener('submit', handleChordGeneration);
    rhymeForm.addEventListener('submit', handleRhymeSearch);
    imageryForm.addEventListener('submit', handleImagerySearch);
    searchForm.addEventListener('submit', handleSongSearch);
    saveButton.addEventListener('click', handleSaveSong);
    
    // Load saved songs on startup
    loadSavedSongs();
});

// API Functions
async function fetchFromAPI(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        return { error: error.message };
    }
}

async function pollTaskStatus(taskId) {
    return new Promise((resolve, reject) => {
        const checkStatus = async () => {
            try {
                const result = await fetchFromAPI(`/task/${taskId}`);
                
                if (result.error) {
                    reject(result.error);
                    return;
                }
                
                if (result.status === 'completed') {
                    resolve(result.result);
                    return;
                }
                
                if (result.status === 'error') {
                    reject(result.error);
                    return;
                }
                
                // Still pending, check again after interval
                setTimeout(checkStatus, POLLING_INTERVAL);
            } catch (error) {
                reject(error);
            }
        };
        
        checkStatus();
    });
}

// Event Handlers
async function handleLyricsGeneration(event) {
    event.preventDefault();
    
    const artist = artistInput.value;
    const title = titleInput.value;
    const theme = themeInput.value;
    const lyricsPrompt = lyricsPromptInput.value;
    const temperature = parseFloat(temperatureInput.value);
    
    // Format the prompt
    let prompt = '';
    if (artist) prompt += `<|artist|>${artist}<|/artist|>\n`;
    if (title) prompt += `<|title|>${title}<|/title|>\n`;
    if (theme) prompt += `Theme: ${theme}\n`;
    if (lyricsPrompt) prompt += lyricsPrompt;
    
    // Show loading state
    workspaceLyrics.value = 'Generating lyrics...';
    
    try {
        // Send request to generate lyrics
        const response = await fetchFromAPI('/generate/lyrics', 'POST', {
            prompt,
            temperature,
            max_length: 500,
            num_return_sequences: 1
        });
        
        if (response.error) {
            throw new Error(response.error);
        }
        
        // Poll for results
        const lyrics = await pollTaskStatus(response.task_id);
        
        // Update workspace
        workspaceTitle.value = title || 'Untitled';
        workspaceLyrics.value = lyrics[0]; // Use the first generated sequence
    } catch (error) {
        console.error('Lyrics generation failed:', error);
        workspaceLyrics.value = `Error generating lyrics: ${error.message}`;
    }
}

async function handleChordGeneration(event) {
    event.preventDefault();
    
    const key = keyInput.value;
    const section = sectionInput.value;
    
    // Show loading state
    chordResult.textContent = 'Generating chord progression...';
    
    try {
        // Send request to generate chord progression
        const response = await fetchFromAPI('/generate/chord_progression', 'POST', {
            key,
            section_type: section,
            temperature: 0.7
        });
        
        if (response.error) {
            throw new Error(response.error);
        }
        
        // Poll for results
        const chords = await pollTaskStatus(response.task_id);
        
        // Display the chord progression
        chordResult.textContent = chords.join(' - ');
        
        // Add a button to insert into lyrics
        const insertButton = document.createElement('button');
        insertButton.className = 'btn btn-sm btn-outline-primary mt-2';
        insertButton.textContent = 'Insert into lyrics';
        insertButton.onclick = () => {
            const chordText = `[${section.toUpperCase()} - ${key}]\n${chords.join(' ')}\n\n`;
            workspaceLyrics.value = chordText + workspaceLyrics.value;
        };
        
        chordResult.appendChild(document.createElement('br'));
        chordResult.appendChild(insertButton);
    } catch (error) {
        console.error('Chord generation failed:', error);
        chordResult.textContent = `Error generating chord progression: ${error.message}`;
    }
}

async function handleRhymeSearch(event) {
    event.preventDefault();
    
    const word = rhymeWordInput.value.trim();
    if (!word) return;
    
    // Show loading state
    rhymeResults.innerHTML = '<div class="text-center">Searching for rhymes...</div>';
    
    try {
        // Send request to find rhymes
        const response = await fetchFromAPI('/find/rhymes', 'POST', {
            word,
            num_rhymes: 20
        });
        
        if (response.error) {
            throw new Error(response.error);
        }
        
        // Poll for results
        const rhymes = await pollTaskStatus(response.task_id);
        
        // Display the rhymes
        rhymeResults.innerHTML = '';
        
        if (rhymes.length === 0) {
            rhymeResults.innerHTML = '<div class="text-center">No rhymes found</div>';
            return;
        }
        
        rhymes.forEach(rhyme => {
            const rhymeItem = document.createElement('span');
            rhymeItem.className = 'rhyme-item';
            rhymeItem.textContent = rhyme;
            rhymeItem.onclick = () => {
                // Insert at cursor position or at the end
                insertAtCursor(workspaceLyrics, rhyme);
            };
            
            rhymeResults.appendChild(rhymeItem);
        });
    } catch (error) {
        console.error('Rhyme search failed:', error);
        rhymeResults.innerHTML = `<div class="text-center text-danger">Error: ${error.message}</div>`;
    }
}

async function handleImagerySearch(event) {
    event.preventDefault();
    
    const theme = imageryThemeInput.value.trim();
    if (!theme) return;
    
    // Show loading state
    imageryResults.innerHTML = '<div class="text-center">Searching for imagery...</div>';
    
    try {
        // Send request to suggest imagery
        const response = await fetchFromAPI('/suggest/imagery', 'POST', {
            theme,
            num_suggestions: 10
        });
        
        if (response.error) {
            throw new Error(response.error);
        }
        
        // Poll for results
        const suggestions = await pollTaskStatus(response.task_id);
        
        // Display the imagery suggestions
        imageryResults.innerHTML = '';
        
        if (suggestions.length === 0) {
            imageryResults.innerHTML = '<div class="text-center">No suggestions found</div>';
            return;
        }
        
        suggestions.forEach(suggestion => {
            const imageryItem = document.createElement('span');
            imageryItem.className = 'imagery-item';
            imageryItem.textContent = suggestion;
            imageryItem.onclick = () => {
                // Insert at cursor position or at the end
                insertAtCursor(workspaceLyrics, suggestion);
            };
            
            imageryResults.appendChild(imageryItem);
        });
    } catch (error) {
        console.error('Imagery search failed:', error);
        imageryResults.innerHTML = `<div class="text-center text-danger">Error: ${error.message}</div>`;
    }
}

async function handleSongSearch(event) {
    event.preventDefault();
    
    const artist = searchArtistInput.value.trim();
    const title = searchTitleInput.value.trim();
    
    if (!artist || !title) {
        searchResults.innerHTML = '<div class="text-center text-warning">Please enter both artist and title</div>';
        return;
    }
    
    // Show loading state
    searchResults.innerHTML = '<div class="text-center">Searching for song...</div>';
    
    try {
        // Send request to fetch song
        const song = await fetchFromAPI('/fetch/song', 'POST', {
            artist,
            title
        });
        
        if (song.error) {
            throw new Error(song.error);
        }
        
        // Display the song
        searchResults.innerHTML = `
            <div class="search-result">
                <div class="search-result-title">${song.title}</div>
                <div class="search-result-artist">by ${song.artist}</div>
                <div class="search-result-lyrics">${song.lyrics}</div>
                <button class="btn btn-sm btn-outline-primary mt-2" id="use-as-reference">Use as Reference</button>
            </div>
        `;
        
        // Add event listener to the "Use as Reference" button
        document.getElementById('use-as-reference').addEventListener('click', () => {
            // Add to workspace as a reference
            const referenceText = `REFERENCE: "${song.title}" by ${song.artist}\n\n`;
            workspaceLyrics.value = referenceText + workspaceLyrics.value;
        });
    } catch (error) {
        console.error('Song search failed:', error);
        searchResults.innerHTML = `<div class="text-center text-danger">Error: ${error.message}</div>`;
    }
}

function handleSaveSong() {
    const title = workspaceTitle.value.trim() || 'Untitled';
    const lyrics = workspaceLyrics.value.trim();
    
    if (!lyrics) {
        alert('Please add some lyrics before saving');
        return;
    }
    
    // Create song object
    const song = {
        id: Date.now().toString(),
        title,
        lyrics,
        date: new Date().toISOString()
    };
    
    // Save to local storage
    const savedSongs = JSON.parse(localStorage.getItem('savedSongs') || '[]');
    savedSongs.push(song);
    localStorage.setItem('savedSongs', JSON.stringify(savedSongs));
    
    // Update the list
    loadSavedSongs();
    
    alert('Song saved successfully!');
}

function loadSavedSongs() {
    const savedSongs = JSON.parse(localStorage.getItem('savedSongs') || '[]');
    
    savedSongsList.innerHTML = '';
    
    if (savedSongs.length === 0) {
        savedSongsList.innerHTML = '<tr><td colspan="3" class="text-center">No saved songs</td></tr>';
        return;
    }
    
    savedSongs.forEach(song => {
        const row = document.createElement('tr');
        
        // Format date
        const date = new Date(song.date);
        const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
        
        row.innerHTML = `
            <td>${song.title}</td>
            <td>${formattedDate}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary load-song" data-id="${song.id}">Load</button>
                <button class="btn btn-sm btn-outline-danger delete-song" data-id="${song.id}">Delete</button>
            </td>
        `;
        
        savedSongsList.appendChild(row);
    });
    
    // Add event listeners
    document.querySelectorAll('.load-song').forEach(button => {
        button.addEventListener('click', () => {
            const songId = button.getAttribute('data-id');
            const savedSongs = JSON.parse(localStorage.getItem('savedSongs') || '[]');
            const song = savedSongs.find(s => s.id === songId);
            
            if (song) {
                workspaceTitle.value = song.title;
                workspaceLyrics.value = song.lyrics;
            }
        });
    });
    
    document.querySelectorAll('.delete-song').forEach(button => {
        button.addEventListener('click', () => {
            if (!confirm('Are you sure you want to delete this song?')) return;
            
            const songId = button.getAttribute('data-id');
            let savedSongs = JSON.parse(localStorage.getItem('savedSongs') || '[]');
            savedSongs = savedSongs.filter(s => s.id !== songId);
            localStorage.setItem('savedSongs', JSON.stringify(savedSongs));
            
            loadSavedSongs();
        });
    });
}

// Helper Functions
function insertAtCursor(textarea, text) {
    const startPos = textarea.selectionStart;
    const endPos = textarea.selectionEnd;
    const scrollTop = textarea.scrollTop;
    
    textarea.value = textarea.value.substring(0, startPos) + text + 
                     textarea.value.substring(endPos, textarea.value.length);
    
    textarea.focus();
    textarea.selectionStart = startPos + text.length;
    textarea.selectionEnd = startPos + text.length;
    textarea.scrollTop = scrollTop;
}

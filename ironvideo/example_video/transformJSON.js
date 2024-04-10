const video_url_path = "./";
const run_data_path = "./test3_short1/";

function transformURL(url) {
    url = url.replace("https://face-shame-proj.s3.il-central-1.amazonaws.com/source-vid/",run_data_path);
    url = url.replace("https://vid.israeltechguard.org/source-vid/",run_data_path);
    
    return url;
}

function transformJSON(json) {
  return new Promise((resolve, reject) => {
    // Create a new script element
    var script = document.createElement('script');
    
    // Set the src attribute to the URL of your external script
    script.src = run_data_path + "main_json.js";
    
    // Set the async attribute to false to block execution until the script is loaded
    script.async = false;
    
    // Define an event handler for the script's onload event
    script.onload = function() {
      console.log('Script loaded successfully');
      
      var video_url = main_json.video_url;
      video_url = video_url.replace("https://face-shame-proj.s3.il-central-1.amazonaws.com/source-vid/",video_url_path);
      video_url = video_url.replace("https://vid.israeltechguard.org/source-vid/",video_url_path);
      main_json.video_url = video_url;
      
      for ( var id in main_json.persons ) {      
        main_json.persons[id].main_image = run_data_path + "person_" + id + "/main_picture/main_pic.jpg";
        //main_json.persons[id].main_image_50 = transformURL(main_json.persons[id].main_image_50);
        //main_json.persons[id].main_image_100 = transformURL(main_json.persons[id].main_image_100);
        //main_json.persons[id].main_image_200 = transformURL(main_json.persons[id].main_image_200);
      }
      
      resolve(main_json);      
    };
    
    // Define an event handler for the script's onerror event in case the script fails to load
    script.onerror = function() {
      console.error('Script failed to load');
    };
    
    // Append the script element to the document's head or body
    document.head.appendChild(script);
  });
}
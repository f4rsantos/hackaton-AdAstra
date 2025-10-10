import streamlit as st
import zipfile
import io
import os
from PIL import Image
from functions import pil_to_cv, clean_template_name, load_stored_templates


def delete_template_from_disk(template_data):
    """Delete a template file from disk storage.
    
    Args:
        template_data: Template data dictionary containing class_folder and original_name
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        templates_dir = "stored_templates"
        class_folder = template_data.get('class_folder', None)
        original_name = template_data.get('original_name', None)
        
        if not original_name:
            return False, "No original filename found"
        
        # Construct file path
        if class_folder:
            filepath = os.path.join(templates_dir, class_folder, original_name)
        else:
            filepath = os.path.join(templates_dir, original_name)
        
        # Delete file if it exists
        if os.path.exists(filepath):
            os.remove(filepath)
            return True, f"Deleted {filepath}"
        else:
            return False, f"File not found: {filepath}"
            
    except Exception as e:
        return False, f"Error deleting file: {str(e)}"


def render():
    """Render the Template Management tab"""
    st.markdown('<h2 class="sub-header">📁 Drone Pattern Templates</h2>', unsafe_allow_html=True)
    
    # Add button to reload templates from disk (fixes accumulated duplicates)
    col_reload, col_info = st.columns([1, 3])
    with col_reload:
        if st.button("🔄 Reload from Disk", help="Reload templates from stored_templates/ folder"):
            st.session_state.templates = load_stored_templates()
            # Reinitialize template_folders
            st.session_state.template_folders = {'Default': {}}
            if st.session_state.templates:
                for key, value in st.session_state.templates.items():
                    st.session_state.template_folders['Default'][key] = value
            st.session_state.active_folder = 'Default'
            st.success(f"✅ Reloaded {len(st.session_state.templates)} templates from disk")
            st.rerun()
    
    with col_info:
        st.info("💾 Templates are stored in `stored_templates/<class>/` folders")
    
    # Initialize session state if needed
    if 'templates' not in st.session_state or st.session_state.templates is None:
        st.session_state.templates = {}
    if 'template_folders' not in st.session_state:
        st.session_state.template_folders = {'Default': {}}
    if 'active_folder' not in st.session_state:
        st.session_state.active_folder = 'Default'
    
    # Template Folder Management
    st.markdown("### 📂 Template Folder Management")
    col_folder1, col_folder2, col_folder3 = st.columns([2, 1, 1])
    
    with col_folder1:
        # Active folder selection
        folder_names = list(st.session_state.template_folders.keys())
        current_folder_index = folder_names.index(st.session_state.active_folder) if st.session_state.active_folder in folder_names else 0
        
        selected_folder = st.selectbox(
            "Active Template Folder",
            folder_names,
            index=current_folder_index,
            help="Select the active template folder to work with"
        )
        
        if selected_folder != st.session_state.active_folder:
            st.session_state.active_folder = selected_folder
            # Update templates to show current folder contents
            st.session_state.templates = st.session_state.template_folders[selected_folder].copy()
            st.rerun()
    
    with col_folder2:
        # Create new folder
        new_folder_name = st.text_input("New Folder Name", placeholder="Enter folder name...")
        if st.button("➕ Create Folder") and new_folder_name:
            if new_folder_name not in st.session_state.template_folders:
                st.session_state.template_folders[new_folder_name] = {}
                st.success(f"Created folder: {new_folder_name}")
                st.rerun()
            else:
                st.warning(f"Folder '{new_folder_name}' already exists!")
    
    with col_folder3:
        # Rename current folder
        if st.session_state.active_folder != 'Default':
            rename_folder = st.text_input("Rename Active Folder", value=st.session_state.active_folder)
            if st.button("✏️ Rename Folder") and rename_folder != st.session_state.active_folder:
                if rename_folder not in st.session_state.template_folders:
                    # Rename the folder
                    st.session_state.template_folders[rename_folder] = st.session_state.template_folders[st.session_state.active_folder]
                    del st.session_state.template_folders[st.session_state.active_folder]
                    st.session_state.active_folder = rename_folder
                    st.success(f"Renamed folder to: {rename_folder}")
                    st.rerun()
                else:
                    st.warning(f"Folder '{rename_folder}' already exists!")
        
        # Delete current folder (except Default)
        if st.session_state.active_folder != 'Default' and len(st.session_state.template_folders) > 1:
            if st.button("🗑️ Delete Folder", help="Delete the active folder and all its templates"):
                del st.session_state.template_folders[st.session_state.active_folder]
                st.session_state.active_folder = 'Default'
                st.session_state.templates = st.session_state.template_folders['Default'].copy()
                st.success("Folder deleted!")
                st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Upload Templates to '{st.session_state.active_folder}' Folder**")
        
        # Upload method selection
        upload_type = st.radio(
            "Upload Method",
            ["Individual Files", "Folder/ZIP Archive"],
            horizontal=True,
            help="Choose whether to upload individual template files or a folder/ZIP containing templates"
        )
        
        if upload_type == "Individual Files":
            uploaded_templates = st.file_uploader(
                "Upload drone pattern templates (PNG/JPG)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload known drone spectrogram patterns to use as detection templates"
            )
            
            if uploaded_templates:
                for template_file in uploaded_templates:
                    try:
                        # Ensure templates is initialized as dict
                        if not isinstance(st.session_state.templates, dict):
                            st.session_state.templates = {}
                        
                        # Load and process template
                        template_image = Image.open(template_file)
                        template_cv = pil_to_cv(template_image)
                        
                        # Extract folder name from filename (e.g., "nowe-A.png" -> folder "nowe")
                        filename = template_file.name
                        name_without_ext = os.path.splitext(filename)[0]
                        
                        # Extract folder from name (before dash or use as-is)
                        if '-' in name_without_ext:
                            folder_name = name_without_ext.rsplit('-', 1)[0]  # "nowe-A" -> "nowe"
                        else:
                            folder_name = name_without_ext
                        
                        # Save to disk in appropriate folder
                        templates_dir = "stored_templates"
                        class_dir = os.path.join(templates_dir, folder_name)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        filepath = os.path.join(class_dir, filename)
                        template_image.save(filepath, 'PNG')
                        
                        # Clean the template name for display
                        clean_name = clean_template_name(filename)
                        
                        # Create unique key including folder
                        template_key = f"{folder_name}/{filename}"
                        
                        # Store in both session templates and folder structure
                        template_data = {
                            'image': template_cv,
                            'pil_image': template_image,
                            'size': template_image.size,
                            'clean_name': f"{folder_name}/{clean_name}",
                            'original_name': filename,
                            'class_folder': folder_name
                        }
                        
                        st.session_state.templates[template_key] = template_data
                        st.session_state.template_folders[st.session_state.active_folder][template_key] = template_data
                        
                    except Exception as e:
                        st.error(f"Error loading template {template_file.name}: {str(e)}")
        
        else:  # Folder/ZIP Archive
            uploaded_folder = st.file_uploader(
                "Upload ZIP file containing template images",
                type=['zip'],
                help="Upload a ZIP file containing template images to add to the current folder"
            )
            
            if uploaded_folder:
                try:
                    with zipfile.ZipFile(uploaded_folder, 'r') as zip_ref:
                        template_files = [f for f in zip_ref.namelist() 
                                        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX/')]
                        
                        if template_files:
                            st.success(f"✅ Found {len(template_files)} template images in ZIP")
                            
                            for template_file_path in template_files:
                                try:
                                    with zip_ref.open(template_file_path) as f:
                                        img_content = f.read()
                                        template_image = Image.open(io.BytesIO(img_content))
                                        template_cv = pil_to_cv(template_image)
                                        
                                        # Ensure templates is initialized as dict
                                        if not isinstance(st.session_state.templates, dict):
                                            st.session_state.templates = {}
                                        
                                        # Get just the filename from the path
                                        filename = template_file_path.split('/')[-1]
                                        clean_name = clean_template_name(filename)
                                        
                                        # Store in both session templates and folder structure
                                        template_data = {
                                            'image': template_cv,
                                            'pil_image': template_image,
                                            'size': template_image.size,
                                            'clean_name': clean_name,
                                            'original_name': filename
                                        }
                                        
                                        st.session_state.templates[filename] = template_data
                                        st.session_state.template_folders[st.session_state.active_folder][filename] = template_data
                                        
                                except Exception as e:
                                    st.error(f"Error loading template from ZIP {template_file_path}: {str(e)}")
                        else:
                            st.warning("⚠️ No template image files found in ZIP")
                            
                except Exception as e:
                    st.error(f"❌ Error reading ZIP file: {str(e)}")
    
    with col2:
        st.markdown("**Template Library**")
        
        # Show folder summary
        total_templates = sum(len(folder) for folder in st.session_state.template_folders.values())
        templates_count = len(st.session_state.templates) if st.session_state.templates else 0
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📂 Folder Overview</h4>
            <p><strong>Total Folders:</strong> {len(st.session_state.template_folders)}</p>
            <p><strong>Total Templates:</strong> {total_templates}</p>
            <p><strong>Active Folder:</strong> {st.session_state.active_folder}</p>
            <p><strong>Templates in Active:</strong> {templates_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.templates:
            st.markdown(f"""
            <div class="template-box">
                <h4>📚 Templates in '{st.session_state.active_folder}'</h4>
                <p><strong>Count:</strong> {len(st.session_state.templates)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display templates from active folder, grouped by class
            # Group templates by class folder
            templates_by_class = {}
            root_templates = {}
            
            for name, template_data in st.session_state.templates.items():
                class_folder = template_data.get('class_folder', None)
                if class_folder:
                    if class_folder not in templates_by_class:
                        templates_by_class[class_folder] = {}
                    templates_by_class[class_folder][name] = template_data
                else:
                    root_templates[name] = template_data
            
            # Display root-level templates first (if any)
            if root_templates:
                st.markdown("**📁 Root Templates**")
                for name, template_data in root_templates.items():
                    clean_display_name = template_data.get('clean_name', clean_template_name(name))
                    with st.expander(f"🎯 {clean_display_name}"):
                        col_img, col_info = st.columns([2, 1])
                        
                        with col_img:
                            st.image(template_data['pil_image'], 
                                    caption=f"Template: {clean_display_name}", 
                                    width='stretch')
                        
                        with col_info:
                            st.write(f"**Size:** {template_data['size'][0]} x {template_data['size'][1]}")
                            st.write(f"**File:** {name}")
                            st.write(f"**Folder:** {st.session_state.active_folder}")
                            st.write(f"**Storage:** `stored_templates/{name}`")
                            
                            # Move to different folder
                            other_folders = [f for f in st.session_state.template_folders.keys() if f != st.session_state.active_folder]
                            if other_folders:
                                move_to_folder = st.selectbox(f"Move to folder", other_folders, key=f"move_{name}")
                                if st.button(f"📁 Move", key=f"move_btn_{name}"):
                                    # Move template to selected folder
                                    st.session_state.template_folders[move_to_folder][name] = template_data
                                    del st.session_state.templates[name]
                                    del st.session_state.template_folders[st.session_state.active_folder][name]
                                    st.success(f"Moved {clean_display_name} to {move_to_folder}")
                                    st.rerun()
                            
                            if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                                # Delete from disk
                                success, msg = delete_template_from_disk(template_data)
                                if success:
                                    st.success(f"✅ {msg}")
                                else:
                                    st.warning(f"⚠️ {msg}")
                                
                                # Remove from session state (check if exists first)
                                if name in st.session_state.templates:
                                    del st.session_state.templates[name]
                                if st.session_state.active_folder in st.session_state.template_folders:
                                    if name in st.session_state.template_folders[st.session_state.active_folder]:
                                        del st.session_state.template_folders[st.session_state.active_folder][name]
                                st.rerun()
            
            # Display class-organized templates
            for class_name in sorted(templates_by_class.keys()):
                class_templates = templates_by_class[class_name]
                st.markdown(f"**📂 Class: {class_name}** ({len(class_templates)} templates)")
                
                for name, template_data in class_templates.items():
                    clean_display_name = template_data.get('clean_name', clean_template_name(name))
                    with st.expander(f"🎯 {clean_display_name}"):
                        col_img, col_info = st.columns([2, 1])
                        
                        with col_img:
                            # Handle templates with either pil_image or just image (from Train Mode)
                            if 'pil_image' in template_data:
                                display_image = template_data['pil_image']
                            elif 'image' in template_data:
                                # Convert OpenCV to PIL for display
                                from functions import cv_to_pil
                                display_image = cv_to_pil(template_data['image'])
                            else:
                                st.error("No image data found")
                                continue
                            
                            st.image(display_image, 
                                    caption=f"Template: {clean_display_name}", 
                                    width='stretch')
                        
                        with col_info:
                            # Get size from either pil_image or image
                            if 'size' in template_data:
                                size = template_data['size']
                            elif 'image' in template_data:
                                img_shape = template_data['image'].shape
                                size = (img_shape[1], img_shape[0])  # width, height
                            else:
                                size = ("?", "?")
                            
                            st.write(f"**Size:** {size[0]} x {size[1]}")
                            st.write(f"**File:** {template_data.get('original_name', name)}")
                            st.write(f"**Class:** {class_name}")
                            st.write(f"**Folder:** {st.session_state.active_folder}")
                            st.write(f"**Storage:** `stored_templates/{class_name}/{template_data.get('original_name', name)}`")
                            
                            # Move to different folder
                            other_folders = [f for f in st.session_state.template_folders.keys() if f != st.session_state.active_folder]
                            if other_folders:
                                move_to_folder = st.selectbox(f"Move to folder", other_folders, key=f"move_{name}")
                                if st.button(f"📁 Move", key=f"move_btn_{name}"):
                                    # Move template to selected folder
                                    st.session_state.template_folders[move_to_folder][name] = template_data
                                    del st.session_state.templates[name]
                                    del st.session_state.template_folders[st.session_state.active_folder][name]
                                    st.success(f"Moved {clean_display_name} to {move_to_folder}")
                                    st.rerun()
                            
                            if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                                # Delete from disk
                                success, msg = delete_template_from_disk(template_data)
                                if success:
                                    st.success(f"✅ {msg}")
                                else:
                                    st.warning(f"⚠️ {msg}")
                                
                                # Remove from session state (check if exists first)
                                if name in st.session_state.templates:
                                    del st.session_state.templates[name]
                                if st.session_state.active_folder in st.session_state.template_folders:
                                    if name in st.session_state.template_folders[st.session_state.active_folder]:
                                        del st.session_state.template_folders[st.session_state.active_folder][name]
                                st.rerun()
        else:
            st.info("👆 Upload template patterns to get started")

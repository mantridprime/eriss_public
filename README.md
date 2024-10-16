<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

this is a fork of llama 3

an interceptor function was created to inject a hardcoded thought prompt, run a generation and then add the throughts to the response prompt

the llama files will be renamed when i make a PR to huggingface

this current version is a WIP and has debug logging until i verify this code as it was copied from a full featured experimental version
